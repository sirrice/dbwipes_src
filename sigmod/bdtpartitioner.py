import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from collections import deque
from itertools import chain

from rtree.index import Index as RTree
from rtree.index import Property as RProp
from learners.cn2sd.rule import fill_in_rules
from learners.cn2sd.refiner import *
from score import QuadScoreSample7
from bottomup.bounding_box import *
from bottomup.cluster import *
from util import *

from util import *
from basic import Basic
from sampler import Sampler
from merger import Merger
from settings import *

inf = 1e10000000
_logger = get_logger()




class Node(object):
    def __init__(self, rule):
        self.rule = rule
        self.children = []
        self.parent = None
        self.n = 0
        self.influence = -inf
        self.prev_attr = None
        self.score = inf


        self.cards = None # caches the cardinality of in each input group
        self.states = None # caches M-tuples

    def set_score(self, score):
        self.influence = score

    def add_child(self, child):
        self.children.append(child)


    def __leaves__(self):
        if not self.children:
            return [self]
        return chain(*[child.leaves for child in self.children])
    leaves = property(__leaves__)

    def __str__(self):
        return '%.4f\t%d\t%s' % (self.influence, self.n, self.rule)


class BDTTablesPartitioner(Basic):

    def set_params(self, **kwargs):
        Basic.set_params(self, **kwargs)

        self.p = kwargs.get('p', 0.6)
        self.tau = kwargs.get('tau', [0.001, 0.15])
        self.epsilon = kwargs.get('epsilon', 0.005)
        self.min_pts = 5
        self.SCORE_ID = kwargs['SCORE_ID']
        self.inf_bounds = [inf, -inf]
        self.min_improvement = kwargs.get('min_improvement', .01)

        self.sampler = Sampler(self.SCORE_ID)


    def setup_tables(self, tables, merged):
        self.merged_table = merged
        self.tables = tables
        self.err_funcs = [self.err_func.clone() for t in tables]

        for ef, t in zip(self.err_funcs, self.tables):
            ef.setup(t)

#        self.SCORE_ID = add_meta_column(
#                chain(tables, [merged]),
#                SCORE_VAR
#        )
#
        self.sampler = Sampler(self.SCORE_ID)
        self.samp_rates = [best_sample_size(len(t), self.epsilon)/(float(len(t))+1) for t in self.tables]


    def __call__(self, tables, full_table, **kwargs):
        self.setup_tables(tables, full_table)
        base_rule = SDRule(full_table, None)
        node = Node(base_rule)
        tree = self.grow(node, self.tables, self.samp_rates)
        return tree.leaves

    def sample(self, data, samp_rate):
        return self.sampler(data, samp_rate)


    def should_stop(self, samples): 
        def f(args):
            idx, sample = args
            if len(sample) < self.min_pts:
                return True
            h = lambda row: self.influence(row, idx)
            infmax = max(map(h, sample))
            thresh = self.compute_threshold(infmax)
            n, mean, std, maxv, minv = self.compute_score(sample, idx)
            return std < thresh and maxv - minv < thresh

        return reduce(and_, map(f, enumerate(samples)))


    def influence(self, row, idx):
       if row[self.SCORE_ID].value == -inf:
            influence = self.err_funcs[idx]((row,))
            row[self.SCORE_ID] = influence
            self.inf_bounds[0] = min(influence, self.inf_bounds[0])
            self.inf_bounds[1] = max(influence, self.inf_bounds[1])
       return row[self.SCORE_ID].value

    def compute_threshold(self, infmax):
        infl, infu = tuple(self.inf_bounds)
        tau, p = self.tau, self.p
        s = (tau[0] - tau[1]) / ((1-p)*infu - p * infl)
        w = tau[0] + s*(infmax - infu)
        w = min(tau[1], w)
        return w * (infu - infl)       


    def estimate_infs(self, samples):
        f = lambda (idx, sample): map(lambda row: self.influence(row, idx), sample)
        infs = map(f, enumerate(samples))
        infs = filter(bool, infs)
        return infs

    def estimate_inf(self, samples):
        return np.mean(map(np.mean,self.estimate_infs(samples)))

    def get_scores(self, rules, samples):
        scores = []
        f = lambda sample: self.get_score_for_rules(rules, sample)
        scores = map(f, samples)
        scores = filter(lambda s: s!=-inf, scores)
        return scores

    def get_score(self, rules, samples):
        scores = self.get_scores(rules, samples)
        score = scores and self.merge_scores(scores) or -inf
        return score

    def get_score_for_rules(self, rules, sample):
        new_samples = map(lambda r: r.filter_table(sample), rules)
        return self.get_score_for_samples(new_samples)

    def get_score_for_samples(self, samples):
        f = lambda (idx, sample): self.compute_score(sample, idx)
        samples = filter(lambda s: len(s), samples)
        if len(samples) > 1:
            stats = map(f, enumerate(samples))
            scores = []
            counts = []
            for n, mean, std, maxinf, mininf in stats:
                thresh = self.compute_threshold(maxinf)
                scores.append( ((thresh - self.inf_bounds[0]) / (self.inf_bounds[1]-self.inf_bounds[0])) * (std - thresh) )
                counts.append(n)
            return np.mean(scores)#wmean(scores, counts)#ret
                
            args = stats[0] + stats[1] + (0.05,)
            return welchs_ttest(*args)
        return -inf




    def compute_score(self, data, idx):
        f = lambda row: self.influence(row, idx)
        try:
            infs = map(f, data)
            return len(infs), wmean(infs, infs), np.std(infs), max(infs), min(infs)
        except:
            pdb.set_trace()


    def merge_scores(self, scores):
        if scores:
            return np.mean(scores)
        return -inf

    def print_status(self, rule, samples):
        all_infs = self.estimate_infs(samples)
        maxstd = max(map(np.std, all_infs))
        minmin = min(map(min, all_infs))
        uu = np.mean(map(np.mean, all_infs))
        maxmax = max(map(max, all_infs))
        minthresh = self.compute_threshold(maxmax)
        _logger.debug( '%s\t%4f - %4f\t%.4f\t%.4f : %.4f : %.4f\t%.4f\t%s' , self.should_stop(samples), self.inf_bounds[0], self.inf_bounds[1], maxstd, minmin, uu, maxmax, minthresh, str(rule))

 
    def in_box(self, node):
        xb, yb = False, False
        for c in node.rule.filter.conditions:
            attr = self.merged_table.domain[c.position]
            if attr.name == 'x':
                if c.min >= 35 and c.max <= 65:
                    xb = True
            elif attr.name == 'y':
                if c.min >= 35 and c.max <= 65:
                    yb = True
        return xb and yb


    def get_states(self, tables):#node):
        #tables = map(node.rule,self.tables)

        # find tuples in each table that is closest to the average influence
        f = lambda (idx, tables): map(lambda row: self.influence(row, idx), tables)
        all_infs = map(f, enumerate(tables))
        states = []

        for idx, t, infs in zip(xrange(len(tables)), tables, all_infs):
            if infs:
                avg = np.mean(infs)
                min_tup = min(t, key=lambda row: self.influence(row, idx))
                state = self.err_funcs[idx].state((min_tup,))
                states.append(state)
            else:
                states.append(None)

        return states
    
    def skinny_penalty(self, rules):
        for rule in rules:
            edges = []
            for c in rule.filter.conditions:
                attr = self.merged_table.domain[c.position]
                if attr.var_type == Orange.feature.Type.Discrete:
                    continue
                edges.append(c.max - c.min)
            if len(edges) > 1:
                volume = reduce(mul, edges)
                mean_edge = sum(edges) / float(len(edges))
                max_vol = mean_edge ** len(edges)
                perc = (volume / max_vol) ** (1./len(edges))
                if perc < 0.05:
                    return True
                return (1. - perc) * 1.5
            return False

        
        
    def grow(self, node, tables, samp_rates):
        rule = node.rule
        data = rule.examples
        datas = map(rule.filter_table, tables)
        samples = [self.sample(*pair) for pair in zip(datas, samp_rates)]
        counts = map(len, datas)
        node.n = sum(counts)
        node.cards = counts

        if node.n == 0:
            return node

        self.print_status(rule, samples)

        if self.should_stop(samples):
            node.set_score(self.estimate_inf(samples))
            node.states = self.get_states(datas)
            if not node.states:
                pdb.set_trace()
                self.get_states(data)
            return node


        attr_scores = []
        for attr, new_rules in self.child_rules(rule):
            scores = self.get_scores(new_rules, samples)
            if not scores:
                continue
            score = self.merge_scores(scores)
            # penalize excessive splitting along a single dimension if it is not helping
            if attr == node.prev_attr:
                score = score + (0.05) * abs(score)
            if attr.var_type == Orange.feature.Type.Discrete:
                score = score - (0.05) * abs(score)
            if attr == node.prev_attr and self.skinny_penalty(new_rules):
                score = score + (0.6) * abs(score)

            if score != -inf:
                attr_scores.append((attr, new_rules, score, scores))

        if not attr_scores:
            node.set_score(self.estimate_inf(samples))
            node.states = self.get_states(datas)
            return node

        attr, new_rules, score, scores = min(attr_scores, key=lambda p: p[-2])
        


        node.score = min(scores)
        minscore = self.get_score_for_samples(samples)
        minscore = minscore - abs(minscore) * self.min_improvement
        if min(scores) >= minscore and minscore != -inf:
            _logger.debug("score didn't improve\t%.7f >= %.7f", min(scores), minscore)
            node.set_score(self.estimate_inf(samples))
            return node

        all_new_srs = self.update_sample_rates(new_rules, datas, samp_rates)

        for new_rule, new_samp_rates in zip(new_rules, all_new_srs):
            child = Node(new_rule)
            child.prev_attr = attr
            child.parent = node

            self.grow(child, datas, new_samp_rates)

            if child and child.n:
                if child.influence != -inf:
                    node.add_child(child)

        if len(node.children):
            node.set_score(max([n.influence for n in node.children]))
        else:
            node.set_score(self.estimate_inf(samples))
        return node


    def child_rules(self, rule, attrs=None):
        attrs = attrs or self.cols
        next_rules = defaultdict(list)
        cont_attrs = [attr.name for attr in self.merged_table.domain if attr.name in attrs and attr.var_type != Orange.feature.Type.Discrete]
        dist_attrs = [attr.name for attr in self.merged_table.domain if attr.name in attrs and attr.var_type == Orange.feature.Type.Discrete]

        if cont_attrs:
            refiner = BeamRefiner(attrs=cont_attrs, fanout=2)
            for attr, new_rule in refiner(rule):
                next_rules[attr].append(new_rule)
        if dist_attrs:
            refiner = BeamRefiner(attrs=dist_attrs, fanout=5)
            for attr, new_rule in refiner(rule):
                next_rules[attr].append(new_rule)
        return next_rules.items()


        
    def update_sample_rates(self, rules, tables, srs):
        srs_by_table = [[0]*len(srs) for i in tables]
        for idx, (t, samp_rate) in enumerate(zip(tables, srs)):
            if not samp_rate:
                continue
            new_tables = [r.filter_table(t) for r in rules]
            if not sum(map(len, new_tables)):
                continue
            new_samp_rates = self.update_sample_rates_helper(new_tables, samp_rate, idx)
            srs_by_table[idx] = new_samp_rates
        return zip(*srs_by_table)

    def update_sample_rates_helper(self, datas, samp_rate, idx):
        influences, counts = [], []
        f = lambda row: self.influence(row, idx) - self.inf_bounds[0]
        for data in datas:
            influence = sum(map(f, data))
            influences.append(influence)
            counts.append(len(data)+1.)

        total_inf = sum(influences)
        total_count = sum(counts)
        if not total_inf:
            return [0]*len(datas)
        samp_rates = []
        nsamples = total_count * samp_rate
        for influence, count in zip(influences, counts):
            infr = influence / total_inf
            sub_samples = infr * nsamples
            nsr = sub_samples / count
            nsr = min(1., nsr)
            samp_rates.append(nsr)

        return samp_rates


       
        



