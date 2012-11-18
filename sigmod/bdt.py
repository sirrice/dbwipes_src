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
from bottomup.bounding_box import *
from bottomup.cluster import *
from errfunc import ErrTypes
from util import *

from util import *
from basic import Basic
from sampler import Sampler
from merger import Merger
from settings import *
from bdtpartitioner import *

inf = 1e10000000
_logger = get_logger()








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


        self.bad_states = [ef.state(t) for ef, t in zip(self.bad_err_funcs, self.bad_tables)]
        self.good_states = [ef.state(t) for ef, t in zip(self.good_err_funcs, self.good_tables)]


    def nodes_to_clusters(self, nodes, table):
        clusters = []
        for node in nodes:
            node.rule.quality = node.influence
            fill_in_rules((node.rule,), table, cols=self.cols)
            cluster = Cluster.from_rule(node.rule, self.cols)
            cluster.states = node.states
            cluster.cards = node.cards
            clusters.append(cluster)
        return clusters

    def estimate_influence(self, cluster):
        print 'dawg'
        bad_infs, good_infs = [], []

        for ef, big_state, state, n in zip(self.bad_states, cluster.bad_states, cluster.bad_cards):
            if not state:
                continue
            inf = ef.recover(ef.remove(big_state, state, n))
            bad_infs.append(inf)
        if not bad_infs:
            return -1e100000000
        
        if cluster.good_states:
            for ef, big_state, state, n in zip(self.good_states, cluster.good_states, cluster.good_cards):
                inf = ef.recover(ef.remove(big_state, state))
                good_infs.append(inf)
        

        return self.l * np.mean(bad_infs) - (1. - self.l) * max(map(abs, good_infs))
        

    def merge(self, clusters, thresh):
        start = time.time()
        is_mergable = lambda c: c.error >= thresh
        params = dict(self.params)
        params.update({'cols' : self.cols,
                      'err_func' : self.err_func,
                      'influence' : lambda c: self.influence_cluster(c),
                      'is_mergable' : is_mergable,
                      'learner' : self})
        self.merger = Merger(**params)
        merged_clusters = self.merger(clusters)
        merged_clusters.sort(key=lambda c: c.error, reverse=True)
        self.merge_cost = time.time() - start
        _logger.debug("----merge costs----")
        _logger.debug( "merge cost\t%d" , self.merge_cost)
        stats = sorted(self.merger.stats.items(), key=lambda s: s[1], reverse=True)
        strs = ['%s\t%.4f\t%d\t%.4f' % (k, v[0], v[1], v[1] and v[0]/v[1] or 0) for k,v in stats]
        _logger.debug('\n%s', '\n'.join(strs))

        for func, cost in self.merger.stats.items():
            self.costs['merge_%s'%func] = cost
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
        bqueue = deque(bclusters)
        low_influence = []
#        low_influence = [c for c in bclusters if c.error < u]
#        bqueue = deque([c for c in bclusters if c.error >= u])

        hclusters = [c for c in hclusters if c.error >= u]
        if not hclusters:
            for c in bclusters:
                c.bad_states = c.states
                c.bad_cards = c.cards
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
                c.bad_states = c.states
                c.bad_cards = c.cards
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
                    c.good_states = hc.states
                    c.bad_states = c.states
                    c.bad_cards = c.cards
                    c.good_cards = [math.ceil(n * c.volume / hc.volume) for n in c.cards]
                    if not c.good_cards:
                        pdb.set_trace()
                    ret.append(c)
                    continue
                else:
                    for cluster in chain(intersects, excludes):
                        cluster.good_inf, cluster.bad_inf, cluster.error = hc.error, c.error, c.error
                        cluster.states = c.states
                        new_vol = cluster.volume
                        cluster.cards = [math.ceil(n * new_vol / c.volume) for n in c.cards]

                    bqueue.extendleft(intersects)
                    bqueue.extendleft(excludes)
                    break

            if not matched:
                c.bad_inf = c.error
                c.good_inf = -1e10000000
                c.bad_states = c.states
                c.bad_cards = c.cards
                ret.append(c)

        _logger.info( "intersection %d untouched, %d split" , len(low_influence), len(ret))
        ret.extend(low_influence)
        return ret


    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        params = dict(self.params)
        params.update(kwargs)
        params['SCORE_ID'] = self.SCORE_ID

        start = time.time()
        bpartitioner = BDTTablesPartitioner(**params)
        bnodes = list(bpartitioner(bad_tables, full_table))
        #bnodes = list(bpartitioner([full_table], full_table))
        bnodes.sort(key=lambda n: n.influence, reverse=True)
        bclusters = self.nodes_to_clusters(bnodes, full_table)
        self.cost_partition_bad = time.time() - start
        

        _logger.debug('\npartitioning bad tables done\n')

        start = time.time()
        err_func = params['aggerr'].error_func.clone()
        err_func.errtype = ErrTypes(ErrTypes.EQUALTO)
        params['err_func'] = err_func
        hpartitioner = BDTTablesPartitioner(**params)
        hpartitioner.inf_bounds = bpartitioner.inf_bounds
        hnodes = list(hpartitioner(good_tables, full_table))
        hnodes.sort(key=lambda n: n.influence, reverse=True)
        hclusters = self.nodes_to_clusters(hnodes, full_table)
        self.cost_partition_good = time.time() - start



        _logger.debug( "==== Best Bad Clusters (%d total) ====" , len(bnodes))
        _logger.debug( '\n'.join(map(str, bnodes[:10])))
        _logger.debug( "==== Best Good Clusters (%d total) ====" , len(hnodes))
        _logger.debug( '\n'.join(map(str, hnodes[:10])))


        start = time.time()
        _logger.debug('intersecting')
        clusters = self.intersect(bclusters, hclusters)
        self.all_clusters = clusters
        _logger.debug('done in %d', time.time()-start)
        self.cost_split = time.time() - start


        start = time.time()
        _logger.debug("computing influences on %d", len(clusters))
        clusters = filter(lambda c: c.error != -1e10000000, clusters)
        clusters.sort(key=lambda c: c.error, reverse=True)
        thresh = compute_clusters_threshold(clusters, nstds=1.5)
        _logger.debug('computed influences  in %d', time.time()-start)


        _logger.debug('merging')
        self.final_clusters = self.merge(clusters, thresh)        
        self.final_clusters = filter(lambda c: not math.isinf(c.error) and not math.isnan(c.error), self.final_clusters)
        self.cost_merge = time.time() - start


        self.costs.update( {'cost_partition_bad' : self.cost_partition_bad,
                'cost_partition_good' : self.cost_partition_good,
                'cost_split' : self.cost_split,
                'cost_merge' : self.cost_merge})
        
        return self.final_clusters

