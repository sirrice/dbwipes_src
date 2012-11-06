import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from itertools import chain


from learners.cn2sd.rule import fill_in_rules
from learners.cn2sd.refiner import *
from score import QuadScoreSample7
from bottomup.bounding_box import *
from bottomup.cluster import *
from util import *

from util.misc import powerset
from basic import Basic
from sampler import Sampler
from merger import Merger
from settings import *

inf = 1e10000000



class Node(object):
    def __init__(self, rule):
        self.rule = rule
        self.children = []
        self.influence = -inf

    def set_score(self, score):
        self.influence = score

    def add_child(self, child):
        self.children.append(child)

    def __n__(self):
        return len(self.rule.examples)
    n = property(__n__)

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
        self.samp_rate = 1.
        self.SCORE_ID = kwargs['SCORE_ID']
        self.inf_bounds = [inf, -inf]

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
#        (tables,), merged_table = reconcile_tables(tables)
        self.setup_tables(tables, full_table)
        base_rule = SDRule(full_table, None)
        tree = self.grow(base_rule, self.tables, self.samp_rates)
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
            thresh = self.compute_threshold(infmax,  self.inf_bounds[0], self.inf_bounds[1])
            score = self.compute_score(sample, idx)
            return score < thresh

        return reduce(and_, map(f, enumerate(samples)))


    def influence(self, row, idx):
       if row[self.SCORE_ID].value == -inf:
            influence = self.err_funcs[idx]((row,))
            row[self.SCORE_ID] = influence
            self.inf_bounds[0] = min(influence, self.inf_bounds[0])
            self.inf_bounds[1] = max(influence, self.inf_bounds[1])
       return row[self.SCORE_ID].value

    def compute_threshold(self, infmax, infl, infu):
        tau, p = self.tau, self.p
        s = (tau[0] - tau[1]) / ((1-p)*infu - p * infl)
        w = tau[0] + s*(infu - infmax)
        w = min(tau[1], w)
        return w * (infu - infl)       



    def estimate_inf(self, samples):
        f = lambda (idx, sample): map(lambda row: self.influence(row, idx), sample)
        infs = map(f, enumerate(samples))
        infs = filter(bool, infs)
        ret = np.mean(map(np.mean,infs))
        return ret

    def get_score(self, rules, samples):
        scores = []
        for sample in samples:
            new_samples = map(lambda r: r.filter_table(sample), rules)
            f = lambda (idx, sample): self.compute_score(sample, idx)
            score = sum(map(f, enumerate(new_samples)))
            scores.append(score)
        score = self.merge_scores(scores)
        return score

    def compute_score(self, data, idx):
        f = lambda row: self.influence(row, idx)
        return np.std(map(f, data))


    def merge_scores(self, scores):
        return min(scores)
 
        
    def grow(self, rule, tables, samp_rates):
        data = rule.examples
        datas = map(rule.filter_table, tables)
        samples = [self.sample(*pair) for pair in zip(datas, samp_rates)]
        node = Node(rule)

        if sum(map(len, datas)) == 0:
            return None

        print map(len, datas), samp_rates, self.estimate_inf(samples), rule

        if self.should_stop(samples):
            node.set_score(self.estimate_inf(samples))
            return node

        scores = []
        for attr, new_rules in self.child_rules(rule):
            score = self.get_score(new_rules, samples)
            scores.append((new_rules, score))

        if not scores:
            node.set_score(self.estimate_inf(samples))
            return node

        new_rules, score = min(scores, key=lambda p: p[1])

        try:
            all_new_srs = self.update_sample_rates(new_rules, datas, samp_rates)
        except:
            pdb.set_trace()
            all_new_srs = self.update_sample_rates(new_rules, datas, samp_rates)
        for new_rule, new_samp_rates in zip(new_rules, all_new_srs):
            child = self.grow(new_rule, datas, new_samp_rates)
            if child and child.n:
                node.add_child(child)

        return node


    def child_rules(self, rule, attrs=None):
        attrs = attrs or self.cols
        next_rules = defaultdict(list)
        refiner = BeamRefiner(attrs=attrs, fanout=2)
        for attr, new_rule in refiner(rule):
            next_rules[attr].append(new_rule)
        return next_rules.items()


        
    def update_sample_rates(self, rules, tables, srs):
        srs_by_table = []
        for idx, (t, samp_rate) in enumerate(zip(tables, srs)):
            new_tables = [r.filter_table(t) for r in rules]
            if not sum(map(len, new_tables)):
                srs_by_table.append([0]*len(srs))
                continue
            new_samp_rates = self.update_sample_rates_helper(new_tables, samp_rate, idx)
            srs_by_table.append(new_samp_rates)
        return zip(*srs_by_table)

    def update_sample_rates_helper(self, datas, samp_rate, idx):
        influences, counts = [], []
        f = lambda row: self.influence(row, idx) - self.inf_bounds[0]
        for data in datas:
            influence = sum(map(f, data))
            influences.append(influence)
            counts.append(float(len(data)))

        total_inf = sum(influences)
        total_count = sum(counts)
        samp_rates = []
        nsamples = total_count * samp_rate
        for influence, count in zip(influences, counts):
            infr = influence / total_inf
            sub_samples = infr * nsamples
            nsr = sub_samples / count
            nsr = min(1., nsr)
            samp_rates.append(nsr)

#        if sum(samp_rates) != samp_rate:
#            print samp_rates, samp_rate
#            pdb.set_trace()

        return samp_rates


       
        


    def get_splits(self, rule, sample):
        infavg = np.mean(map(self.influence, sample))
        splits = Splits(infavg)
        
        if self.should_stop(sample):
            return splits

        for attr, new_rules in self.child_rules(rule):
            new_samples = map(lambda r: r.filter_table(sample), new_rules)
            score = sum(map(self.compute_score, enumerate(new_samples)))
            splits.add_split(Split(attr, new_rules, score))

        return splits






class BDT(Basic):

    def __init__(self, **kwargs):
        Basic.__init__(self, **kwargs)

    def setup_tables(self, full_table, bad_tables, good_tables, **kwargs):
        Basic.setup_tables(self, full_table, bad_tables, good_tables, **kwargs)

        self.SCORE_ID = add_meta_column(
                chain([full_table], bad_tables, good_tables),
                SCORE_VAR
        )

    def nodes_to_clusters(self, nodes, table):
        rules = []
        for node in nodes:
            rules.append(node.rule)
            node.rule.quality = node.influence
        fill_in_rules(rules, table, cols=self.cols)
        clusters = [Cluster.from_rule(rule, self.cols) for rule in rules]
        return clusters 


    def merge(self, clusters):
#        start = time.time()
#        params = dict(self.params)
#        params.update({'cols' : self.cols,
#                       'full_table' : bdtp.merged_table,
#                       'bad_tables' : bdtp.tables,
#                       'good_tables' : [],
#                       'bad_err_funcs' : bdtp.err_funcs,
#                       'good_err_funcs' : [],
#                       'err_func' : self.err_func})
#        pdb.set_trace()
#        self.merger = ReexecMerger(**params)
#        self.final_clusters = self.merger(self.all_clusters)
#        self.final_clusters.sort(key=lambda c: c.error, reverse=True)
#        self.merge_cost = time.time() - start
        return clusters

    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        params = dict(self.params)
        params['SCORE_ID'] = self.SCORE_ID
        bdtp = BDTTablesPartitioner(**params)



        nodes = list(bdtp(bad_tables, full_table))
        nodes.sort(key=lambda n: n.influence, reverse=True)
        clusters = self.nodes_to_clusters(nodes, full_table)

        self.final_clusters = self.merge(clusters)        

        
        return self.final_clusters

