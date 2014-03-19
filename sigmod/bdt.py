import json
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
from rangemerger import RangeMerger, get_frontier
from settings import *
from bdtpartitioner import *

inf = float('inf')
_logger = get_logger()








class BDT(Basic):

    def __init__(self, **kwargs):
        Basic.__init__(self, **kwargs)
        self.all_clusters = []
        self.cost_split = 0.
        self.cost_partition_bad = 0.
        self.cost_partition_good = 0.
        self.cache = None
        self.use_mtuples = kwargs.get('use_mtuples', False)
        self.max_wait = kwargs.get('max_wait', None)


    def __hash__(self):
        components = [
                self.__class__.__name__,
                str(self.aggerr.__class__.__name__),
                str(set(self.cols)),
                self.epsilon,
                self.tau,
                self.p,
                self.err_func.__class__.__name__,
                self.tablename,
                self.aggerr.keys,
                self.max_wait
                ]
        components = map(str, components)
        return hash('\n'.join(components))
 

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
            rules = [node.rule]
            #rules = [r.simplify() for r in rules]
            fill_in_rules(rules, table, cols=self.cols)
            cluster = Cluster.from_rule(rules[0], self.cols)
            cluster.states = node.states
            cluster.cards = node.cards
            clusters.append(cluster)
        return clusters

    def nodes_to_popular_clusters(self, nodes, table):
      if not nodes: return []
      from collections import Counter
      counter = Counter()
      str_to_rule = {}

      for node in nodes:
        r = node.rule
        if len(r.filter.conditions) > 1:
          for cond in r.filter.conditions:
            newr = SDRule(r.data, r.targetClass, [cond], r.g)
            newr.quality = node.influence
            counter[newr] += 1
            str_to_rule[newr] = newr

      thresh = np.percentile(counter.values(), 70)
      rules = []
      for strrule, count in counter.iteritems():
        if count >= thresh:  #0.25 * len(nodes):
          r = str_to_rule[strrule]
          rules.append(r)

      fill_in_rules(rules, table, cols=self.cols)
      clusters = [Cluster.from_rule(r, self.cols) for r in rules]
      return clusters


    def estimate_influence(self, cluster):
        bad_infs, good_infs = [], []

        for ef, big_state, state, n in zip(self.bad_states, cluster.bad_states, cluster.bad_cards):
            if not state:
                continue
            influence = ef.recover(ef.remove(big_state, state, n))
            bad_infs.append(influence)
        if not bad_infs:
            return -inf
        
        if cluster.good_states:
            for ef, big_state, state, n in zip(self.good_states, cluster.good_states, cluster.good_cards):
                influence = ef.recover(ef.remove(big_state, state))
                good_infs.append(influence)
        

        return self.l * np.mean(bad_infs) - (1. - self.l) * max(map(abs, good_infs))


    @instrument
    def merge(self, clusters):
        if len(clusters) <= 1:
            return clusters
        start = time.time()
        clusters.sort(key=lambda c: c.error, reverse=True)
        thresh = compute_clusters_threshold(clusters, nstds=1.5)
        is_mergable = lambda c: c.error >= thresh
        if [attr for attr in self.full_table.domain if attr.varType == orange.VarTypes.Discrete]:
            use_mtuples = self.use_mtuples
            #use_mtuples = False
        else:
            use_mtuples = self.use_mtuples
        params = dict(self.params)
        params.update({
          'cols' : self.cols,
          'influence' : lambda c: self.influence_cluster(c),
          'is_mergable' : is_mergable,
          'use_mtuples' : use_mtuples,
          'c_range': self.c_range,
          'learner' : self
        })
        self.merger = RangeMerger(**params)
        #self.merger = Merger(**params)
        merged_clusters = self.merger(clusters)
        self.merge_cost = time.time() - start

        merged_clusters.sort(key=lambda c: c.error, reverse=True)
        self.merge_stats(self.merger.stats, 'merge_')
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

    
    @instrument
    def intersect(self, bclusters, hclusters):
        errors = [c.error for c in bclusters]
        u, std = np.mean(errors), np.std(errors)
        u = min(max(errors), u + std)
        bqueue = deque(bclusters)
        low_influence = []
#        low_influence = [c for c in bclusters if c.error < u]
        bqueue = deque([c for c in bclusters if c.error >= u])

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
            hcs = filter(c.discretes_contains, hcs)

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
                c.good_inf = -inf
                c.bad_states = c.states
                c.bad_cards = c.cards
                ret.append(c)

        _logger.info( "intersection %d untouched, %d split" , len(low_influence), len(ret))
        ret.extend(low_influence)
        return ret

    @instrument
    def load_from_cache(self):
        import bsddb as bsddb3
        self.cache = bsddb3.hashopen('./dbwipes.cache')
        try:
            myhash = str(hash(self))
            if myhash in self.cache and self.use_cache:
                dicts, nonleaf_dicts, errors = json.loads(self.cache[myhash])
                clusters = map(Cluster.from_dict, dicts)
                nonleaf_clusters = map(Cluster.from_dict, nonleaf_dicts)
                for err, c in zip(errors, chain(clusters, nonleaf_clusters)):
                  c.error = err
                return clusters, nonleaf_clusters
        except:
            pass
        finally:
            self.cache.close()
        return None, None

    @instrument
    def cache_results(self, clusters, nonleaf_clusters):
        import bsddb as bsddb3
        # save the clusters in a dictionary
        if self.use_cache:
            myhash = str(hash(self))
            self.cache = bsddb3.hashopen('./dbwipes.cache')
            try:
                dicts = [c.to_dict() for c in clusters]
                nonleaf_dicts = [c.to_dict() for c in nonleaf_clusters]
                errors = [c.error for c in chain(clusters, nonleaf_clusters)]
                self.cache[myhash] = json.dumps((dicts, nonleaf_dicts, errors))
            except:
                pass
            finally:
                self.cache.close()



    @instrument
    def get_partitions(self, full_table, bad_tables, good_tables, **kwargs):
        clusters, nonleaf_clusters = self.load_from_cache()
        if clusters:
            return clusters, nonleaf_clusters


        params = dict(self.params)
        params.update(kwargs)
        params['SCORE_ID'] = self.SCORE_ID
        params['err_funcs'] = self.bad_err_funcs
        max_wait = params.get('max_wait', None)
        if max_wait:
          params['max_wait'] = max_wait * 2. / 3.


        start = time.time()
        bpartitioner = BDTTablesPartitioner(**params)
        bnodes = list(bpartitioner(bad_tables, full_table))
        bnodes = list(bpartitioner.root.leaves)
        bnodes.sort(key=lambda n: n.influence, reverse=True)
        _logger.debug("bad nodes --> clusters")
        bclusters = self.nodes_to_clusters(bnodes, full_table)
        self.cost_partition_bad = time.time() - start
        

        _logger.debug("clone badnodes")
        htree = bpartitioner.root.clone()
        for hnode in htree.nodes:
          hnode.frombad = True
        _logger.debug('\npartitioning bad tables done\n')
        inf_bound = [inf, -inf]
        for ib in bpartitioner.inf_bounds:
          inf_bound[0] = min(ib[0], inf_bound[0])
          inf_bound[1] = max(ib[1], inf_bound[1])

        start = time.time()
        params['err_funcs'] = self.good_err_funcs
        if isinstance(max_wait, int):
          params['max_wait'] = max_wait / 3.
        hpartitioner = BDTTablesPartitioner(**params)
        hpartitioner.inf_bounds = [list(inf_bound) for t in good_tables]
        hnodes = list(hpartitioner(good_tables, full_table, root=htree))
        hnodes = list(hpartitioner.root.leaves)
        hnodes = filter(lambda n: not n.frombad, hnodes)
        hnodes.sort(key=lambda n: n.influence, reverse=True)
        hclusters = self.nodes_to_clusters(hnodes, full_table)
        self.cost_partition_good = time.time() - start

        _logger.debug( "==== Best Bad Clusters (%d total) ====" , len(bnodes))
        _logger.debug( '\n'.join(map(str, bnodes[:10])))
        _logger.debug( "==== Best Good Clusters (%d total) ====" , len(hnodes))
        _logger.debug( '\n'.join(map(str, hnodes[:10])))


        start = time.time()
        clusters = self.intersect(bclusters, hclusters)
        nonleaves = []
        nonleaves.extend(bpartitioner.root.nonleaves)
        popular_clusters = self.nodes_to_popular_clusters(nonleaves, full_table)
        nonleaf_clusters = self.nodes_to_clusters(
            nonleaves,
            full_table
        )
        nonleaf_clusters = filter_bad_clusters(nonleaf_clusters)
        self.cost_split = time.time() - start

        clusters = filter_bad_clusters(clusters)
        clusters.extend(popular_clusters)

        start = time.time()
        for c in chain(clusters, nonleaf_clusters):
          c.error = self.influence_cluster(c)
        self.stats['init_cluster_errors'] = [time.time()-start, 1]


        self.cache_results(clusters, nonleaf_clusters)

        self.merge_stats(bpartitioner.stats)
        self.merge_stats(hpartitioner.stats)


        return clusters, nonleaf_clusters




    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        clusters, nomerge_clusters = self.get_partitions(full_table, bad_tables, good_tables, **kwargs)
        self.all_clusters = clusters

        for c in nomerge_clusters:
          c.inf_func = c.create_inf_func(self.l)
          c.c_range = list(self.c_range)


        start = time.time()
        _logger.debug('merging')
        final_clusters = self.merge(clusters)        
        final_clusters.extend(nomerge_clusters)
        final_clusters = filter_bad_clusters(final_clusters)

        # if running range mergen
        final_clusters, rms = get_frontier(final_clusters)

        self.final_clusters = final_clusters
        self.cost_merge = time.time() - start


        self.costs.update({
          'cost_partition_bad' : self.cost_partition_bad,
          'cost_partition_good' : self.cost_partition_good,
          'cost_split' : self.cost_split,
          'cost_merge' : self.cost_merge
        })
        
        _logger.debug("=== Costs ===")
        for key, stat in sorted(self.stats.items()):
          _logger.debug("%.4f\t%d\t%s", stat[0], stat[1], key)

        return self.final_clusters

