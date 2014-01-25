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
from node import Node

inf = 1e10000000
_logger = get_logger()





class BDTTablesPartitioner(Basic):

    def set_params(self, **kwargs):
        Basic.set_params(self, **kwargs)

        self.p = kwargs.get('p', 0.6)
        self.tau = kwargs.get('tau', [0.001, 0.15])
        self.epsilon = kwargs.get('epsilon', 0.005)
        self.min_pts = 5
        self.SCORE_ID = kwargs['SCORE_ID']
        self.inf_bounds = None 
        self.min_improvement = kwargs.get('min_improvement', .01)


        self.max_wait = kwargs.get('max_wait', 2*60*60)
        self.start_time = None

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

        self.inf_bounds = [[inf, -inf] for table in tables]


    def __call__(self, tables, full_table, **kwargs):
        self.setup_tables(tables, full_table)
        base_rule = SDRule(full_table, None)
        node = Node(base_rule)
        tree = self.grow(node, self.tables, self.samp_rates)
        return tree.leaves

    def sample(self, data, samp_rate):
        return self.sampler(data, samp_rate)

    def should_idx_stop(self, args):
      idx, infs = args
      if len(infs) < self.min_pts:
          return True

      infmax = max(infs)
      thresh = self.compute_threshold(infmax, idx)
      std = np.std(infs)
      maxv, minv = max(infs), min(infs)
      return std < thresh and maxv - minv < thresh

    def print_status(self, rule, datas, sample_infs):
      bools = map(self.should_idx_stop, enumerate(sample_infs))
      perc_passed = np.mean(map(float, bools))

      all_infs = filter(bool, sample_infs)
      maxstd = max(map(np.std, all_infs))
      minmin = min(map(min, all_infs))
      uu = np.mean(map(np.mean, all_infs))
      maxes = map(max, all_infs)
      threshes = [self.compute_threshold(v, idx) for idx, v in enumerate(maxes)]
      npts = sum(map(len, datas)) if datas else 0
      _logger.debug('%.2f\tnpts(%d)\t%4f - %4f\t%.4f\t%.4f : %.4f : %.4f\t%.4f\t%s' , 
                    perc_passed, npts, self.inf_bounds[0][0], self.inf_bounds[0][1], 
                    maxstd, minmin, uu, max(maxes), min(threshes), str(rule))

 
    def should_stop(self, sample_infs): 
      bools = map(self.should_idx_stop, enumerate(sample_infs))
      return reduce(and_, bools)


    def influence(self, row, idx):
       if row[self.SCORE_ID].value == -inf:
            influence = self.err_funcs[idx]((row,))
            row[self.SCORE_ID] = influence
            self.inf_bounds[idx][0] = min(influence, self.inf_bounds[idx][0])
            self.inf_bounds[idx][1] = max(influence, self.inf_bounds[idx][1])
       return row[self.SCORE_ID].value

    def compute_threshold(self, infmax, idx):
        infl, infu = tuple(self.inf_bounds[idx])
        tau, p = self.tau, self.p
        s = (tau[0] - tau[1]) / ((1-p)*infu - p * infl)
        w = tau[0] + s*(infmax - infu)
        w = min(tau[1], w)
        ret = w * (infu - infl)       
        if ret == -inf:
            raise RuntimeError()
        return ret


    def compute_infs(self, idx, samples):
      return [self.influence(r, idx) for r in samples]

    def estimate_inf(self, sample_infs):
        return np.mean(map(np.mean, filter(bool, sample_infs)))

    def get_scores(self, rules, samples):
      sample2infs = defaultdict(list)
      rule2infs = defaultdict(list)
      scores = []
      for idx, sample in enumerate(samples):
        for rule in rules:
          infs = self.compute_infs(idx, rule.filter(sample))
          rule2infs[rule].append(infs)
          sample2infs[idx].append(infs)

      for idx in xrange(len(samples)):
        allinfs = sample2infs[idx]
        score = self.get_score_for_infs([idx]*len(allinfs), allinfs)
        scores.append(score)

      scores = filter(lambda s: s!=-inf, scores)
      return scores

    def get_score_for_infs(self, idxs, samp_infs):
        scores, counts = [], []
        for idx, infs in zip(idxs, samp_infs):
          if not len(infs): continue
          thresh = self.compute_threshold(max(infs), idx)
          bounds = self.inf_bounds[idx]
          inf_range = bounds[1] - bounds[0]
          if not inf_range:
            scores.append(0)
            counts.append(n)
          else:
            std = np.std(infs)
            scores.append(((thresh - bounds[0]) / inf_range) * (std - thresh))
            counts.append(len(infs))
        if scores:
          return np.mean(scores)
        return -inf

    def merge_scores(self, scores):
        if scores:
          return np.percentile(scores, 75)
        return -inf


    def get_states(self, tables):#node):
        #tables = map(node.rule,self.tables)

        # find tuples in each table that is closest to the average influence
        all_infs = []
        for idx, table in enumerate(tables):
            infs = [self.influence(row, idx) for row in table]
            all_infs.append(infs)
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
    
    def adjust_score(self, score, node, attr, rules):
      # penalize excessive splitting along a single dimension if it is not helping
      if attr.var_type == Orange.feature.Type.Discrete:
        score = score - (0.15) * abs(score)
      else:
        if attr == node.prev_attr:
          score = score + (0.01) * abs(score)
          if False and self.skinny_penalty(rules):
            score = score + (0.6) * abs(score)
      return score


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
        node.cards = map(len, datas)
        node.n = sum(node.cards)

        if node.n == 0:
          return node

        f = lambda (idx, samps): self.compute_infs(idx, samps)
        sample_infs = map(f, enumerate(samples))
        curscore = self.get_score_for_infs(range(len(samples)), sample_infs)
        est_inf = self.estimate_inf(sample_infs)
        node.set_score(est_inf)

        if node.parent:
          self.print_status(rule, datas, sample_infs)
          if self.should_stop(sample_infs):
            node.states = self.get_states(datas)
            return node


        attr_scores = []
        for attr, new_rules in self.child_rules(rule):
          scores = self.get_scores(new_rules, samples)
          score = self.merge_scores(scores)
          score = self.adjust_score(score, node, attr, new_rules)
          if score == -inf: continue
          attr_scores.append((attr, new_rules, score, scores))

        if not attr_scores:
          node.states = self.get_states(datas)
          print "no attrscores"
          return node

        attr, new_rules, score, scores = min(attr_scores, key=lambda p: p[-2])

        node.score = min(scores) 
        minscore = curscore - abs(curscore) * self.min_improvement
        if node.score >= minscore and minscore != -inf:
          _logger.debug("score didn't improve\t%.7f >= %.7f", min(scores), minscore)
          return node

        all_new_srs = self.update_sample_rates(new_rules, datas, samp_rates)

        for new_rule, new_samp_rates in zip(new_rules, all_new_srs):
          child = Node(new_rule)
          child.prev_attr = attr
          child.parent = node

          self.grow(child, datas, new_samp_rates)

          if child and child.n and child.influence != -inf:
            node.add_child(child)

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
        f = lambda row: self.influence(row, idx)
        for data in datas:
            influence = map(f, data)
            influence = sum(i - self.inf_bounds[idx][0] for i in influence)
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
            nsr = max(0, min(1., nsr))
            samp_rates.append(nsr)

        return samp_rates


       
        



