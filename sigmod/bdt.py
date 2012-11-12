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




    def nodes_to_clusters(self, nodes, table):
        rules = []
        for node in nodes:
            rules.append(node.rule)
            node.rule.quality = node.influence
        fill_in_rules(rules, table, cols=self.cols)
        clusters = [Cluster.from_rule(rule, self.cols) for rule in rules]
        return clusters 


    def merge(self, clusters, thresh):
        start = time.time()
        #thresh = thresh is None and compute_clusters_threshold(clusters) or thresh
        is_mergable = lambda c: c.error >= thresh
        params = dict(self.params)
        params.update({'cols' : self.cols,
                      'err_func' : self.err_func,
                      'influence' : lambda cluster: self.influence_cluster(cluster),
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
                        cluster.good_inf, cluster.bad_inf, cluster.error = hc.error, c.error, c.error
                    bqueue.extendleft(intersects)
                    bqueue.extendleft(excludes)
                    break

            if not matched:
                c.bad_inf = c.error
                c.good_inf = -1e100000000
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
        #bnodes = list(bpartitioner(bad_tables, full_table))
        bnodes = list(bpartitioner([full_table], full_table))
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
        hnodes = list(hpartitioner([good_tables[0]], full_table))
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
        _logger.debug('done in %d', time.time()-start)
        self.cost_split = time.time() - start


        start = time.time()
        _logger.debug("computing influences on %d", len(clusters))
        clusters = filter(lambda c: c.error != -1e10000000, clusters)
        clusters.sort(key=lambda c: c.error, reverse=True)
        thresh = min(clusters, key=lambda c: c.error).error
        _ = filter(lambda c: c.error >= thresh, clusters)
        n = len(_)
        _ = set(clusters[:n])
        for c in clusters:
            c.error = 0
        thresh = None
        for c in _:
            c.error = self.influence_cluster(c)
            if math.isnan(c.error) or math.isinf(c.error): 
                continue
            thresh = thresh is None and c.error or min(thresh, c.error)
        for c in clusters:
            if c not in _:
               c.error = thresh - 10*thresh
        clusters = filter(lambda c: not math.isinf(c.error), clusters)
        self.all_clusters = clusters
        _logger.debug('computed influences  in %d', time.time()-start)


        _logger.debug('merging')
        self.final_clusters = self.merge(clusters, thresh)        
        self.final_clusters = filter(lambda c: not math.isinf(c.error) and not math.isnan(c.error), self.final_clusters)
        self.cost_merge = time.time() - start


        self.costs = {'cost_partition_bad' : self.cost_partition_bad,
                'cost_partition_good' : self.cost_partition_good,
                'cost_split' : self.cost_split,
                'cost_merge' : self.cost_merge}
        
        return self.final_clusters

