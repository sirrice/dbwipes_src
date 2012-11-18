import math
import pdb
import random
import numpy as np
import sys
import time
sys.path.extend(['.', '..'])

from itertools import chain, repeat
from collections import defaultdict
from rtree.index import Index as RTree
from rtree.index import Property as RProp
from operator import mul, and_, or_

from util import rm_attr_from_domain, get_logger
from util.table import *
from bottomup.bounding_box import *
from bottomup.cluster import *
from zero import Zero
from adjgraph import AdjacencyGraph

_logger = get_logger()

def instrument(fn):
    func_name = fn.__name__
    def w(self, *args, **kwargs):
        start = time.time()
        ret = fn(self, *args, **kwargs)

        if func_name not in self.stats:
            self.stats[func_name] = [0, 0]
        self.stats[func_name][0] += (time.time() - start)
        self.stats[func_name][1] += 1
        return ret
    return w


class Merger(object):
    """
    Merges clusters

    transparently scales
    - cluster bounding boxes by table bounds
    - errors by min/max error
    """

    def __init__(self, **kwargs):
        self.min_clusters = 1
        self.is_mergable = lambda c: c.error > 0 
        self.influence = None
        self.learner = kwargs.get('learner', None)

        self.stats = {}
        self.adj_graph = None
        self.use_mtuples = kwargs.get('use_mtuples', True)
        
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        self.min_clusters = kwargs.get('min_clusters', self.min_clusters)
        self.is_mergable = kwargs.get('is_mergable', self.is_mergable)
        # lambda cluster: influence_value_of(cluster)
        self.influence = kwargs.get('influence', self.influence)
        self.learner = kwargs.get('learner', self.learner)
        self.use_mtuples = kwargs.get('use_mtuples', self.use_mtuples)
        

    def setup_stats(self, clusters):
        """
        computes error bounds and the minimum volume of a 0-volume cluster

        """
        vols = np.array([c.volume for c in clusters if c.volume > 0])
        if len(vols) == 0:
            self.point_volume = 0.00001
        else:
            self.point_volume = vols.min() / 2.


    def setup_errors(self, clusters):
        return

    def scale_error(self, error):
        return (error - self.min_error) / self.diff_error

    def scale_point(self, point):
        return self.zero.zero(np.array(point)).tolist()

    def scale_box(self, box):
        return self.zero.zero(np.array(box)).tolist()

    def filter_discrete(self, c1, c2, intersecting_clusters):
        return filter(c1.discretes_intersect,
                      filter(c2.discretes_intersect, intersecting_clusters))

    def construct_rtree(self, clusters):
        if not len(clusters[0].bbox[0]):
            class k(object):
                def intersection(self, foo):
                    return xrange(len(clusters))
            return k()
        ndim = max(2, len(clusters[0].centroid))
        p = RProp()
        p.dimension = ndim
        p.dat_extension = 'data'
        p.idx_extension = 'index'

        rtree = RTree(properties=p)
        for idx, c in enumerate(clusters):
            box = c.bbox #self.scale_box(c.bbox)
            if ndim == 1:
                rtree.insert(idx, box[0] + [0] + box[1] + [1])
            else:
                rtree.insert(idx, box[0] + box[1])
        return rtree

    @instrument
    def get_intersection(self, rtree, bbox):
        if len(bbox[0]) == 0:
            return rtree.intersection([0, 0, 1, 1])

        #bbox = self.scale_box(bbox)
        if len(bbox[0]) == 1:
            return rtree.intersection(bbox[0] + (0,) + bbox[1] + (1,))
        return rtree.intersection(bbox[0] + bbox[1])        


    def get_states(self, merged, intersecting_clusters):
        @instrument
        def update_states(self, weight, global_states, efs, states, cards):
            if states is None:
                return

            thezip = zip(global_states, efs, states, cards)
            for idx, (gstate, ef, state, n) in enumerate(thezip):
                n = n * weight
                n = int(math.ceil(n))
                if n >= 1:
                    n = int(math.floor(n))
                else:
                    n = random.random() <= n and 1 or 0

                if n and state:
                    ustate = ef.update((state,), n)
                    if not gstate: 
                        global_states[idx] = ustate
                    else:
                        global_states[idx] = ef.update((ustate, gstate))


        bad_states = [None]*len(self.learner.bad_tables)
        good_states = [None]*len(self.learner.good_tables)
        bad_efs = self.learner.bad_err_funcs
        good_efs = self.learner.good_err_funcs
        for inter in intersecting_clusters:
            ibox = intersection_box(inter.bbox, merged.bbox)
            ivol = volume(ibox)
            if ivol < 0:
                continue
            weight = ivol / merged.volume
            update_states(self, weight, bad_states, bad_efs, inter.bad_states, inter.bad_cards)
            update_states(self, weight, good_states, good_efs, inter.good_states, inter.good_cards)

        return bad_states, good_states


    def influence_from_mtuples(self, merged, intersecting_clusters):
        bad_states, good_states = self.get_states(merged, intersecting_clusters)


        if not sum(map(bool, bad_states)):
            return None

        # now compute the influence using these m-tuple states
        @instrument
        def get_influences(self, efs, states, master_states,c ):
            infs = []
            for ef, state, mstate in zip(efs, states, master_states):
                if state:
                    # XXX
                    # XXX: HUGE HACK.  Takes count from state, assumes
                    # XXX: states is m-tuple of avg()
                    # XXX
                    inf = ef.recover(ef.remove(mstate, state)) 
                    inf = inf / (state[-1]**c)  
                    infs.append(inf)
            return infs

        bad_efs = self.learner.bad_err_funcs
        good_efs = self.learner.good_err_funcs
       
        bad_infs = get_influences(self, bad_efs, bad_states, self.learner.bad_states, self.learner.c)
        good_infs = map(abs, get_influences(self, good_efs, good_states, self.learner.good_states, 0) or [])
        if not bad_infs:
            return -1e10000000

        bad_inf = bad_infs and np.mean(bad_infs) or -1e100000000
        good_inf = good_infs and np.mean(good_infs) or 0
        l = self.learner.l
 
        return l * bad_inf - (1. - l) * good_inf

    @instrument
    def merge(self, cluster, neighbor, clusters):
        newbbox = bounding_box(cluster.bbox, neighbor.bbox)
        cidxs = self.get_intersection(self.rtree, newbbox)
        intersecting_clusters = [clusters[cidx] for cidx in cidxs]

        merged = Cluster.merge(cluster, neighbor, intersecting_clusters, self.point_volume)
        if not merged or not merged.volume:
            return None
        
        if self.use_mtuples:
            merged.error = self.influence_from_mtuples(merged, intersecting_clusters)
        else:
            merged.error = self.influence(merged)
        return merged


    def neighbors(self, idxs, matrix):
        ret = set()
        nonzeros = map(np.nonzero, matrix[tuple(idxs),:] == 1)
        map(ret.update, (arr[0].tolist() for arr in nonzeros))
        ret.difference_update(idxs)
        return ret

    @instrument
    def expand(self, cluster, clusters):
        rms = set()
        while True:
            neighbors = self.adj_graph.neighbors(cluster)
            
            rms, tomerges = set(), list()
            for n in neighbors:
                if n in rms or cluster.contains(n) or cluster.same(n, epsilon=0.02):
                    rms.add(n)
                    continue
                tomerges.append(n)

            if not tomerges:
                break

            def filter_cluster(c):
                if c is None:
                    reasons.append('0')
                    return False
                if c.error == None:
                    reasons.append('s')
                    return False
                if c.error == -1e10000000:
                    reasons.append('e')
                    return False
                if math.isnan(c.error):
                    reasons.append('n')
                    return False
                if c.error <= cluster.error:
                    reasons.append('<%.4f'%c.error)
                    return False
                reasons.append('!')
                return True


            tomerges.sort(key=lambda n: volume(bounding_box(cluster.bbox, n.bbox)), reverse=True)
            reasons = []
            merges = []
            seen = []
            f = lambda tomerge: self.merge(cluster, tomerge, clusters)
            for n in tomerges:
                bseen = False
                for _ in seen:
                    if _.contains(n):
                        bseen = True
                if bseen: continue

                _ = f(n)
                if _:
                    seen.append(_)
                if filter_cluster(_):
                    merges.append(_)
                    break
#            merges = map(f, tomerges)
#            merges = filter(filter_cluster, merges)

            _logger.debug("neighbors: %d\t%s", len(neighbors), cluster)
            _logger.debug("reason   :%s", ''.join(reasons))
            if not merges:
                break
            merged = max(merges, key=lambda m: m.error)

            _logger.debug('\tmerged:\t%s',merged)

            rms.update(merged.parents)
            cluster = merged
            self.adj_graph.insert(merged)

        _logger.debug('\n')
        return cluster, rms



    @instrument
    def make_adjacency(self, clusters):
        return AdjacencyGraph(clusters)

        
    def __call__(self, clusters, **kwargs):
        """
        if there is cached info, load it and use the loaded data to
        1) replace structures like adj_graph and rtree
        2) initialize mergeable clusters, etc
        """
        if not clusters:
            return list(clusters)

        _logger.debug("merging %d clusters", len(clusters))

        self.set_params(**kwargs)
        self.setup_stats(clusters)
        clusters_set = set(clusters)
        self.adj_graph = self.make_adjacency(clusters)
        self.rtree = self.construct_rtree(clusters)
        results = []


        mergable_clusters = filter(self.is_mergable, clusters)
        mergable_clusters.sort(key=lambda mc: mc.error, reverse=True)
 
        while len(clusters_set) > self.min_clusters:

            cur_clusters = sorted(clusters_set, key=lambda c: c.error, reverse=True)


            _logger.debug("# mergable clusters: %d\tout of\t%d",
                    len(mergable_clusters),
                    len(cur_clusters))
            if not mergable_clusters:
                break


            map(mergable_clusters[0].adjacent, cur_clusters)

            merged_clusters, new_clusters = set(), set()
            seen = set()

            for cluster in mergable_clusters:
                if cluster in merged_clusters or cluster in new_clusters or cluster in seen:
                    continue

                skip = False
                for test in chain(new_clusters, mergable_clusters):
                    if test == cluster: continue
                    if test.contains(cluster, .01):
                        #_logger.debug("skipped\n\t%s\n\t%s", str(cluster), str(test))
                        skip = True
                        break
                if skip:
                    _logger.debug("skipped\n\t%s\n\t%s", str(cluster), str(test))
                    continue

                merged, rms = self.expand(cluster, clusters) 
                if not merged or merged == cluster or len(filter(lambda c: c.contains(merged), cur_clusters)):
                    seen.add(cluster)
                    continue
                if math.isnan(merged.error) or merged.error == -1e1000000:
                    continue
                
                _logger.debug("%.4f\t%.4f\t-> %.4f",
                               merged.parents[0].error,
                               merged.parents[1].error,
                               merged.error)
                seen.update(merged.parents)
                seen.update(rms)


                if merged not in cur_clusters:
                    new_clusters.add(merged)                    
                merged_clusters.update(rms)

            _logger.debug("merged %d\t%d new clusters\tout of %d",
                          len(merged_clusters),
                          len(new_clusters),
                          len(mergable_clusters))

            
            if not new_clusters:
                break


            map(self.adj_graph.remove, merged_clusters)
            map(self.adj_graph.insert, new_clusters)
            clusters_set.difference_update(merged_clusters)
            clusters_set.update(new_clusters)
            mergable_clusters = sorted(new_clusters, key=lambda c: c.error, reverse=True)

        clusters_set = filter(lambda c: c.error != -1e10000000, clusters_set) 
        return sorted(clusters_set, key=lambda c: c.error, reverse=True)
            
    

