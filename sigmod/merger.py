import json
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

from util import rm_attr_from_domain, get_logger, instrument
from util.table import *
from util.misc import valid_number
from bottomup.bounding_box import *
from bottomup.cluster import *
from zero import Zero
from adjgraph import AdjacencyGraph

_logger = get_logger()


class Merger(object):
    """
    Merges clusters

    transparently scales
    - cluster bounding boxes by table bounds
    - errors by min/max error
    """

    def __init__(self, **kwargs):
        self.min_clusters = 1
        self.influence = None
        self.learner = kwargs.get('learner', None)

        self.stats = defaultdict(lambda: [0, 0])
        self.adj_graph = None
        self.rtree = None
        self.base_clusters = []
        self.use_mtuples = kwargs.get('use_mtuples', True)
        self.use_cache = kwargs.get('use_cache', False)

        # whether or not the partitions cover the entire space (can assume adjacency)
        # False if output from merger
        self.partitions_complete = kwargs.get('partitions_complete', True)

        self.DEBUG = kwargs.get('DEBUG', False)


        self.CACHENAME = './dbwipes.merger.cache'

        self.cache = None# bsddb3.hashopen('./dbwipes.merger.cache')
        
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        self.min_clusters = kwargs.get('min_clusters', self.min_clusters)
        # lambda cluster: influence_value_of(cluster)
        self.influence = kwargs.get('influence', self.influence)
        self.learner = kwargs.get('learner', self.learner)
        self.use_mtuples = kwargs.get('use_mtuples', self.use_mtuples)
        self.use_cache = kwargs.get('use_cache', self.use_cache)
        self.partitions_complete = kwargs.get('partitions_complete', self.partitions_complete)
        

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

    @instrument
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
    def get_intersection(self, bbox):
        if len(bbox[0]) == 0:
            return self.rtree.intersection([0, 0, 1, 1])

        if len(bbox[0]) == 1:
            return self.rtree.intersection(bbox[0] + (0,) + bbox[1] + (1,))
        return self.rtree.intersection(bbox[0] + bbox[1])        


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
                    influence = ef.recover(ef.remove(mstate, state)) 
                    if state[-1]**c:
                        influence = influence / (state[-1]**c)  
                        infs.append(inluencef)
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
      """
      @deprecated!
      """
      newbbox = bounding_box(cluster.bbox, neighbor.bbox)
      cidxs = self.get_intersection(newbbox)
      intersecting_clusters = [clusters[cidx] for cidx in cidxs]
      intersecting_clusters = filter(cluster.discretes_overlap, intersecting_clusters)

      merged = Cluster.merge(cluster, neighbor, intersecting_clusters, self.point_volume)
      if not merged or not merged.volume:
          return None
      
      if self.use_mtuples and cluster.discretes_same(neighbor):
          intersecting_clusters = filter(cluster.discretes_same, intersecting_clusters)
          merged.error = self.influence_from_mtuples(merged, intersecting_clusters)
      else:
          merged.error = self.influence(merged)
      return merged

    @instrument
    def dim_merge(self, cluster, dim, dec=None, inc=None):
      bbox = [list(cluster.bbox[0]), list(cluster.bbox[1])]
      merged = cluster.clone()
      if dec is not None:
        bbox[0][dim] = dec
      if inc is not None:
        bbox[1][dim] = inc
      merged.bbox = (tuple(bbox[0]), tuple(bbox[1]))
      merged.error = self.influence(merged)
      merged.parents = [cluster]
      return merged

    @instrument
    def disc_merge(self, cluster, dim, vals):
      merged = cluster.clone()
      vals = set(vals)
      vals.update(merged.discretes.get(dim, ()))
      merged.discretes[dim] = vals
      merged.error = self.influence(merged)
      merged.parents = [cluster]
      return merged



    def expand_candidates(self, cluster, skip=None):
      """
      Args
        skip    set of clusters that can be skipped
      Yields (dim, dir, g)
            dim: dimention
            dir: direction of expansion ('inc', 'dec', 'disc')
            g: generator of increasing clusters along dim in dir direction
      """
      neighbors = self.adj_graph.neighbors(cluster)
      check = lambda n: not((skip and n in skip) or (n==cluster) or cluster.same(n, epsilon=0.01) or cluster.contains(n))
      neighbors = filter(check, neighbors)
      if not neighbors: return

      f = lambda tomerge: self.merge(cluster, tomerge, clusters)

      # gather all the directions and increments to expand along
      start = time.time()
      dim_to_incs = defaultdict(set)
      dim_to_decs = defaultdict(set)
      dim_to_discs = defaultdict(set)
      for n in neighbors:
        for dim in xrange(len(n.cols)):
          minv, maxv = n.bbox[0][dim], n.bbox[1][dim]
          if minv < cluster.bbox[0][dim]:
            dim_to_decs[dim].add(minv)
          if maxv > cluster.bbox[1][dim]:
            dim_to_incs[dim].add(maxv)

        disc_diffs = cluster.discrete_differences(n, epsilon=0.01)
        if disc_diffs and len(disc_diffs) <= 1:
          for dim, vals in disc_diffs.iteritems():
            dim_to_discs[dim].add(tuple(vals))
      cost = time.time() - start
      self.stats['gather'][0] += cost
      self.stats['gather'][1] += 1

      for dim in xrange(len(cluster.cols)):
        generator = (self.dim_merge(cluster, dim, None, inc) 
            for inc in sorted(dim_to_incs[dim]))
        yield (dim, 'inc', generator)

        generator = (self.dim_merge(cluster, dim, dec, None)
            for dec in sorted(dim_to_decs[dim], reverse=True))
        yield (dim, 'dec', generator)

      for dim, discs in dim_to_discs.iteritems():
        generator = (self.disc_merge(cluster, dim, disc)
            for disc in sorted(discs, key=lambda d: len(d)))
        yield (dim, 'disc', generator)

    def create_filterer(self, cluster, reasons=None):
      def filter_cluster(c, n=None):
        reason = None
        if c is None:
          reason = '.'
        if c.error == None:
          reason = 'N'
        if c.error == -1e10000000 or c.error == 1e1000000000:
          reason = 'e'
        if math.isnan(c.error):
          reason = '_'
        if c.error <= cluster.error:
          reason = '<%.4f'%c.error
        if n and c.error <= n.error:
          reason = '<n'
        ret = reason is not None

        if ret:
          reason = '!'
        if reasons is not None:
          reasons.append(reason)
        return ret
      return filter_cluster


    @instrument
    def expand(self, cluster, clusters):
      rms = set()
      while True:
        reasons = []
        expansions = self.expand_candidates(cluster, rms)
        filter_cluster = self.create_filterer(cluster, reasons)

        merged = cluster
        for dim, direction, g in expansions:
          for cand in g:
            if not filter_cluster(cand): break
            if cand.error > merged.error:
              merged = cand

        if not merged or merged == cluster:
          break

        _logger.debug('\tmerged:\t%s',merged)

        rms.update(merged.parents)
        cluster = merged
        self.adj_graph.insert(merged)

      _logger.debug('\n')
      return cluster, rms



    @instrument
    def cache_results(self, clusters_set, mergable_clusters):
        if not self.use_cache:
            return

        try:
            import bsddb as bsddb3
            self.cache =  bsddb3.hashopen(self.CACHENAME)
            myhash = str(hash(self.learner))
            c = str(self.learner.c)
            key = '%s:%s' % (myhash, c)

            clusters_set = [cluster.to_dict() for cluster in clusters_set]
            mergable_clusters = [cluster.to_dict() for cluster in mergable_clusters]
            self.cache[key] = json.dumps((clusters_set, mergable_clusters))

            cs_to_keys = json.loads(self.cache[myhash]) if myhash in self.cache else {}
            cs_to_keys[c] = key
            self.cache[myhash] = json.dumps(cs_to_keys)
            self.cache.close()
            _logger.debug("saved cache %f", self.learner.c)
        except:
            import traceback
            traceback.print_exc()



    @instrument
    def load_from_cache(self, clusters):
        """
        if there is cached info, load it and use the loaded data to
        1) replace structures like adj_graph and rtree
        2) initialize mergeable clusters, etc

        All state needs to be transient
        - adj_graph
        - rtree
        - clusters list
        - mergeable clusters
        - clusters_set
        
        """
        myhash = str(hash(self.learner))
        c = self.learner.c
        if self.use_cache:
          try:
            import bsddb as bsddb3
            self.cache =  bsddb3.hashopen(self.CACHENAME)
            if myhash not in self.cache:
                self.cache.close()
                raise RuntimeError("cache miss")
            cs_to_keys = json.loads(self.cache[myhash])
            cs_to_keys = dict([(float(k),v) for k,v in cs_to_keys.iteritems()])
            cs = [other_c for other_c in cs_to_keys if other_c >= c]
            key = str(cs_to_keys[min(cs)])
            matches = c == min(cs)

            clusters_set, mergable_clusters = json.loads(self.cache[key])
            clusters_set = set(map(Cluster.from_dict, clusters_set))
            mergable_clusters = map(Cluster.from_dict, mergable_clusters)

            also_mergable = []
            for cluster in clusters:
                useless = False
                for mc in mergable_clusters:
                    useless = useless or mc.contains(cluster)
                    if useless:
                        break
                if not useless:
                    self.adj_graph.insert(cluster)
                    also_mergable.append(cluster)
              
            mergable_clusters.extend(also_mergable)
            clusters_set.update(also_mergable)

            # fix up their error values!
            start = time.time()
            for cluster in chain(mergable_clusters, clusters_set):
              if self.use_mtuples:
                raise RuntimeError("disabled mtuples for now")
                cidxs = self.get_intersection(cluster.bbox)
                intersecting_clusters = [clusters[cidx] for cidx in cidxs]
                cluster.error = self.influence_from_mtuples(cluster, intersecting_clusters)
              else:
                cluster.error = self.influence(cluster)
            self.stats['load_fixup'][0] += time.time()-start
            self.stats['load_fixup'][1] += 1

            self.cache.close()
            alt_mergable = clusters
            if len(alt_mergable) > len(mergable_clusters):
                _logger.debug("loaded from cache\t%.4f", self.learner.c)
                mergable_clusters.sort(key=lambda mc: mc.error, reverse=True)
                return matches, clusters_set, mergable_clusters
            
            alt_mergable.sort(key=lambda mc: mc.error, reverse=True)
            return False, set(clusters), alt_mergable
          except:
            import traceback
            traceback.print_exc()


        clusters_set = set(clusters)
        mergable_clusters.sort(key=lambda mc: mc.error, reverse=True)
        return False, clusters_set, mergable_clusters


    @instrument
    def make_adjacency(self, *args, **kwargs):
        return AdjacencyGraph(*args, **kwargs)

        
    def __call__(self, clusters, **kwargs):
        if not clusters:
            return list(clusters)

        _logger.debug("merging %d clusters", len(clusters))

        self.set_params(**kwargs)
        self.setup_stats(clusters)

        # adj_graph is used to track adjacent partitions
        _logger.debug("building adj graph")
        self.adj_graph = self.make_adjacency(clusters, self.partitions_complete)


        # rtree is static (computed once) to find base partitions within 
        # a merged partition
        #_logger.debug("building rtree")
        #self.rtree = self.construct_rtree(clusters)

        # load state from cache
        can_stop, clusters_set, mergable_clusters = self.load_from_cache(clusters)
        if can_stop:
            return sorted(clusters_set, key=lambda c: c.error, reverse=True)

        _logger.debug("start merging!")
        while len(clusters_set) > self.min_clusters:

          cur_clusters = sorted(clusters_set, key=lambda c: c.error, reverse=True)

          _logger.debug("# mergable clusters: %d\tout of\t%d",
                  len(mergable_clusters),
                  len(cur_clusters))
          if not mergable_clusters:
              break


          merged_clusters, new_clusters = set(), set()
          seen = set()

          for cluster in mergable_clusters:
            if (cluster in merged_clusters or 
                cluster in new_clusters or cluster in seen):
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
            if not valid_number(merged.error):
              continue
            
            _logger.debug("%.4f\t%.4f\t-> %.4f",
                            merged.parents[0].error,
                            merged.parents[0].error,
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

        clusters_set = filter_bad_clusters(clusters_set)
        self.cache_results(clusters_set, mergable_clusters)
        return sorted(clusters_set, key=lambda c: c.error, reverse=True)
            
    

