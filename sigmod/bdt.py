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
            n, mean, std, maxv = self.compute_score(sample, idx)
            return std < thresh

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

    def get_score(self, rules, samples):
        scores = []
        f = lambda sample: self.get_score_for_sample(rules, sample)
        scores = map(f, samples)
        scores = filter(lambda s: s!=-inf, scores)
        score = scores and self.merge_scores(scores) or -inf
        return score

    def get_score_for_sample(self, rules, sample):
        new_samples = map(lambda r: r.filter_table(sample), rules)
        f = lambda (idx, sample): self.compute_score(sample, idx)
        new_samples = filter(lambda s: len(s), new_samples)
        if len(new_samples) > 1:
            stats = map(f, enumerate(new_samples))
            return sum((std - self.compute_threshold(maxinf)) for n, mean, std, maxinf in stats)
                
            args = stats[0] + stats[1] + (0.05,)
            return welchs_ttest(*args)
        return -inf


    def compute_score(self, data, idx):
        f = lambda row: self.influence(row, idx)
        try:
            infs = map(f, data)
            return len(infs), np.mean(infs), np.std(infs), max(infs)
        except:
            pdb.set_trace()


    def merge_scores(self, scores):
        return max(scores)
 
        
    def grow(self, node, tables, samp_rates):
        rule = node.rule
        data = rule.examples
        datas = map(rule.filter_table, tables)
        samples = [self.sample(*pair) for pair in zip(datas, samp_rates)]
        counts = map(len, datas)
        node.n = sum(counts)

        if node.n == 0:
            return node

        all_infs = self.estimate_infs(samples)
        maxstd = max(map(np.std, all_infs))
        minmin = min(map(min, all_infs))
        uu = np.mean(map(np.mean, all_infs))
        maxmax = max(map(max, all_infs))
        minthresh = self.compute_threshold(maxmax)
        print '%s\t%4f - %4f\t%.4f\t%.4f : %.4f : %.4f\t%.4f\t%s' % (self.should_stop(samples), min(self.inf_bounds), max(self.inf_bounds), maxstd, minmin, uu, maxmax, minthresh, str(rule))
        if minthresh < 0:
            pdb.set_trace()
            self.compute_threshold(maxmax)


        if self.should_stop(samples):
            node.set_score(self.estimate_inf(samples))
            return node

        scores = []
        for attr, new_rules in self.child_rules(rule):
            score = self.get_score(new_rules, samples)
            if score != -inf:
                scores.append((attr, new_rules, score))

        if not scores:
            node.set_score(self.estimate_inf(samples))
            return node

        attr, new_rules, score = min(scores, key=lambda p: p[-1])
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
        refiner = BeamRefiner(attrs=attrs, fanout=2)
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


       
        








class BDT(Basic):

    def __init__(self, **kwargs):
        Basic.__init__(self, **kwargs)

    def setup_tables(self, full_table, bad_tables, good_tables, **kwargs):
        Basic.setup_tables(self, full_table, bad_tables, good_tables, **kwargs)


        self.SCORE_ID = add_meta_column(
                chain([full_table], bad_tables, good_tables),
                SCORE_VAR
        )

        domain = self.full_table.domain
        attrnames = [attr.name for attr in domain]
        self.cont_dists = dict(zip(attrnames, Orange.statistics.basic.Domain(self.full_table)))
        self.disc_dists = dict(zip(attrnames, Orange.statistics.distribution.Domain(self.full_table)))




    def nodes_to_clusters(self, nodes, table):
        rules = []
        for node in nodes:
            rules.append(node.rule)
            node.rule.quality = node.influence
        fill_in_rules(rules, table, cols=self.cols)
        clusters = [Cluster.from_rule(rule, self.cols) for rule in rules]
        return clusters 


    def merge(self, clusters):
        start = time.time()
        thresh = compute_clusters_threshold(clusters)
        is_mergable = lambda c: c.error >= 0 #thresh
        params = dict(self.params)
        params.update({'cols' : self.cols,
                      'err_func' : self.err_func,
                      'influence' : lambda cluster: self.influence(cluster),
                      'is_mergable' : is_mergable})
        self.merger = Merger(**params)
        merged_clusters = self.merger(clusters)
        merged_clusters.sort(key=lambda c: c.error, reverse=True)
        self.merge_cost = time.time() - start
        _logger.debug( "merge cost\t%d" , self.merge_cost)
        return merged_clusters

    def create_rtree(self, clusters):
        if not len(clusters[0].bbox[0]):
            class k(object):
                def intersection(self, foo):
                    return xrange(len(clusters))
            return k()

        ndim = len(clusters[0].bbox[0]) + 1
        p = RProp()
        p.dimension = ndim
        p.dat_extension = 'data'
        p.idx_extension = 'index'

        rtree = RTree(properties=p)
        for idx, c in enumerate(clusters):
            rtree.insert(idx, c.bbox[0] + (0,) + c.bbox[1] + (1,))
        return rtree

    
    def intersect(self, bclusters, hclusters):
        errors = [c.error for c in bclusters]
        u, std = np.mean(errors), np.std(errors)
        u = min(max(errors), u + std)
        low_influence = [c for c in bclusters if c.error < u]
        bqueue = deque([c for c in bclusters if c.error >= u])

        hclusters = [c for c in hclusters if c.error >= u]
        if not hclusters:
            return bclusters
        hindex = self.create_rtree(hclusters)
        ret = []


        while bqueue:
            c = bqueue.popleft()

            idxs = hindex.intersection(c.bbox[0] + (0,) + c.bbox[1] + (1,))
            hcs = [hclusters[idx] for idx in idxs]
            hcs = filter(c.discretes_intersect, hcs)

            if not hcs:
                c.bad_inf = c.error
                c.good_inf = None
                ret.append(c)
                continue

            # first duplicate c into clusters that have the same discrete values as hcs
            # and those that are not
            # put the diff clusters back
            matched = False
            for hc in hcs:
                # are they exactly the same? then just skip
                split_clusters = c.split_on(hc)

                if not split_clusters: 
                    # hc was false positive, skip
                    continue

                matched = True

                intersects, excludes = split_clusters
                if len(intersects) == 1 and not excludes:
                    c.good_inf = hc.error
                    c.bad_inf = c.error
                    ret.append(c)
                    continue
                else:
                    for cluster in chain(intersects, excludes):
                        cluster.good_inf, cluster.bad_inf = hc.error, c.error
                    bqueue.extendleft(intersects)
                    bqueue.extendleft(excludes)
                    break

            if not matched:
                c.bad_inf = c.error
                c.good_inf = -1e100000000
                ret.append(c)

        print "intersection %d untouched, %d split" % (len(low_influence), len(ret))
        pdb.set_trace()
        return ret

    def influence(self, cluster):
        rule = cluster.to_rule(self.dummy_table, self.cols, cont_dists=self.cont_dists, disc_dists=self.disc_dists)
        return Basic.influence(self, rule)

    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        params = dict(self.params)
        params['SCORE_ID'] = self.SCORE_ID

        bpartitioner = BDTTablesPartitioner(**params)
        bnodes = list(bpartitioner(bad_tables, full_table))
        bnodes.sort(key=lambda n: n.influence, reverse=True)
        bclusters = self.nodes_to_clusters(bnodes, full_table)

        print

        hpartitioner = BDTTablesPartitioner(**params)
        hpartitioner.inf_bounds = bpartitioner.inf_bounds
        hnodes = list(hpartitioner(good_tables, full_table))
        hnodes.sort(key=lambda n: n.influence, reverse=True)
        hclusters = self.nodes_to_clusters(hnodes, full_table)


        print "==== Best Bad Clusters (%d total) ====" % len(bnodes)
        print '\n'.join(map(str, bnodes[:10]))
        print "==== Best Good Clusters (%d total) ====" % len(hnodes)
        print '\n'.join(map(str, hnodes[:10]))


        start = time.time()
        _logger.debug('intersecting')
        clusters = self.intersect(bclusters, hclusters)
        _logger.debug('done in %d', time.time()-start)
        _logger.debug("computing influences on %d", len(clusters))
        for idx, c in enumerate(clusters):
            c.mean_inf = c.error
        clusters = filter(lambda c: c.mean_inf != -inf, clusters)
        _logger.debug('computed influences  in %d', time.time()-start)

        _logger.debug('merging')
        self.final_clusters = self.merge(clusters)        
        self.all_clusters = clusters
        
        return self.final_clusters

