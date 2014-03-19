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
from scipy.optimize import fsolve

from merger import Merger
from util import rm_attr_from_domain, get_logger, instrument
from util.table import *
from bottomup.bounding_box import *
from bottomup.cluster import *
from zero import Zero
from adjgraph import AdjacencyGraph

_logger = get_logger()


def get_frontier(clusters):
  _logger.debug("get_frontier: %d clusters", len(clusters))
  seen = set()
  rest = set(clusters)
  rms = set()
  while rest:
    #_logger.debug("get_frontier: %d rest\t%d seen", len(rest), len(seen))
    cur = rest.pop()
    if r_empty(cur.c_range):
      rms.add(cur)
      continue
    cur_list = [cur]
    new_cur_list = [cur]


    expanded = set()
    rest_read = set()
    for o in rest:
      rest_read.add(o)
      new_cur_list = []
      for c in cur_list:
        left, right = intersect_c_ranges(c, o)
        new_cur_list.extend(left)
        if not right: rms.add(o)
        expanded.update(right)
      if not new_cur_list: break
      cur_list = new_cur_list

    rest.difference_update(rest_read)
    rest.update(expanded)
    seen.update(new_cur_list)
    if not new_cur_list: rms.add(cur)
  _logger.debug("get_frontier: %d removed, %d result", len(rms), len(seen))

  return seen, rms

def r_vol(bound):
  return bound[1] - bound[0]

def r_empty(bound):
  return bound[1] <= bound[0]

def r_equal(bound1, bound2):
  return bound1[0] == bound2[0] and bound1[1] == bound2[1]

def r_lt(bound1, bound2):
  "bound1 values < bound2 values"
  return bound1[0] < bound2[0] and bound1[1] < bound2[1]

def r_lte(bound1, bound2):
  "bound1 values <= bound2 values"
  return bound1[0] <= bound2[0] and bound1[1] <= bound2[1]


def r_scontains(bound1, bound2):
  "bound1 strictly contains bound2"
  return bound1[0] < bound2[0] and bound2[1] < bound1[1]

def r_contains(bound1, bound2):
  "bound1  contains bound2"
  return bound1[0] <= bound2[0] and bound2[1] <= bound1[1]


def r_intersect(bound1, bound2):
  return [max(bound1[0], bound2[0]), min(bound1[1], bound2[1])]

def r_union(bound1, bound2):
  return [min(bound1[0], bound2[0]), max(bound1[1], bound2[1])]

def r_subtract(bound1, bound2):
  """
  remove bound2 from bound1. 
  Return list of bound
  """
  if r_contains(bound2, bound1):
    return [ [bound1[0], bound1[0]] ]
  if r_scontains(bound1, bound2):
    return [ [bound1[0], bound2[0]], [bound2[1], bound1[1]] ]
  inter = r_intersect(bound1, bound2)
  if r_lte(inter, bound1):
    return [ [inter[1], bound1[1]] ]
  return [ [bound1[0], inter[0]] ]


def binary_search(bound, inf1, inf2):
  roots = fsolve(lambda v: inf1(v) - inf2(v), bound[0])
  roots = filter(lambda v: v >= bound[0] and v <= bound[1], roots)
  if roots:
    print "roots: %s" % (",".join(map(str, roots)))
    return roots[0]

  pdb.set_trace()
  return None



  min1, min2 = (inf1(bound[0]) , inf2(bound[0]))
  max1, max2 = (inf1(bound[1]) , inf2(bound[1]))
  thresh = max(0.00001, (max(max1,max2) - min(min1,min2))/100.)
  i = 0
  while True:
    i += 1
    l = sum(bound) / 2.
    v1, v2 = inf1(l), inf2(l)
    if abs(v1 - v2) < thresh:
      break

    if (min1 < min2) == (v1 < v2):
      bound[0] = l
    else:
      bound[1] = l
  print "roots vs l\t%s\t%.4f" % (','.join(map(str, roots)), l)
  return l


def intersect_c_ranges(cluster1, cluster2):
  """
  return the pairs with updated bounds where they can possibly be
          best
  """
  # if pair1 and pair2 c bounds don't overlap
  #  return them
  c1min, c1max = tuple(cluster1.c_range)
  c2min, c2max = tuple(cluster2.c_range)
  bound1 = cluster1.c_range
  bound2 = cluster2.c_range

  innerbound = r_intersect(bound1, bound2)
  if r_empty(innerbound):
    return [cluster1], [cluster2]


  bound = r_union(bound1, bound2)
  inf1 = cluster1.inf_func
  inf2 = cluster2.inf_func
  min1, min2 = (inf1(bound[0]) , inf2(bound[0]))
  max1, max2 = (inf1(bound[1]) , inf2(bound[1]))
  infb1 = [min1, max1]
  infb2 = [min2, max2]

  ret1, ret2 = [cluster1], [cluster2]
  if not all(map(valid_number, (min1, max1))):
    ret1 = []
  if not all(map(valid_number, (min2, max2))):
    ret2 = []
  if not ret1 or not ret2:
    return ret1, ret2



  roots = fsolve(lambda v: inf1(v)-inf2(v), bound[0])
  roots = filter(lambda v: v >= bound[0] and v <= bound[1], roots)
  if not roots:
    if infb1[0] < infb2[0]:
      cluster1.c_range = [c1min, c1min]
      return [], [cluster2]
    if infb1[0] > infb2[0]:
      cluster2.c_range = [c2min, c2min]
      return [cluster1], []

    print "fuck"
    pdb.set_trace()

  else:
    l = roots[0]
    print "l = %.3f\t%s\t%s" % (l, cluster1, cluster2)

    ret1 = []
    ret2 = []


    if min1 > min2:
      ret1.append([c1min, l])
      ret1.append([c2max, c1max])
      ret2.append([l, c2max])
      ret2.append([c2min, c1min])
    elif min1 < min2:
      ret2.append([c2min, l])
      ret2.append([c1max, c2max])
      ret1.append([c1min, c2min])
      ret1.append([l, c1max])
    elif max1 > max2:
      ret1.append([l, c1max])
      ret1.append([c1min, c2min])
      ret2.append([c2min, l])
      ret2.append([c1max, c2max])
    elif max1 < max2:
      ret1.append([c1min, l])
      ret1.append([c2max, c1max])
      ret2.append([l, c2max])
      ret2.append([c2min, c1min])
    else:
      return [cluster1], [cluster2]

    ret1 = filter(lambda b: b[0] < b[1], ret1)
    ret2 = filter(lambda b: b[0] < b[1], ret2)

    res1 = []
    for b in ret1:
      c = res1 and cluster1.clone() or cluster1
      c.c_range = b
      res1.append(c)
    res2 = []
    for b in ret2:
      c = res2 and cluster2.clone() or cluster2
      c.c_range = b
      res2.append(c)
    return res1, res2
 




  bound = list(innerbound)
  min1, min2 = (inf1(bound[0]) , inf2(bound[0]))
  max1, max2 = (inf1(bound[1]) , inf2(bound[1]))

  ret1, ret2 = [cluster1], [cluster2]
  if not all(map(valid_number, (min1, max1))):
    ret1 = []
  if not all(map(valid_number, (min2, max2))):
    ret2 = []
  if not ret1 or not ret2:
    return ret1, ret2

  infb1 = [min1, max1]
  infb2 = [min2, max2]

    
  # XXX: this is not _quite_ correct
  if r_equal(infb1, infb2):
    return [cluster1], [cluster2]

  elif r_lte(infb1, infb2):
    cluster1.c_range = [c1min, c1min]
    return [], [cluster2]

  elif r_lte(infb2, infb1):
    cluster2.c_range = [c2min, c2min]
    return [cluster1], []

  else:
    l = binary_search(list(bound), inf1, inf2)
    print "l = %.3f\t%s\t%s" % (l, cluster1, cluster2)
    l = max(innerbound[0], l)
    l = min(innerbound[1], l)

   


class RangeMerger(Merger):
  """
  Merges clusters

  transparently scales
  - cluster bounding boxes by table bounds
  - errors by min/max error
  """


  def __init__(self, *args, **kwargs):
    Merger.__init__(self, *args, **kwargs)

    self.c_range = kwargs.get('c_range', [0.01, 0.7])
    self.frontier = set()
    self.CACHENAME = './dbwipes.rangemerger.cache'

  def setup_stats(self, clusters):
    """
    computes error bounds and the minimum volume of a 0-volume cluster

    adds data structures to cluster object
    """
    Merger.setup_stats(self, clusters)

    for c in clusters:
      c.inf_func = c.create_inf_func(self.learner.l)
      c.c_range = list(self.c_range)
      c.inf_range = [c.inf_func(c.c_range[0]), c.inf_func(c.c_range[1])]


  def __call__(self, clusters, **kwargs):
    if not clusters:
        return list(clusters)

    _logger.debug("merging %d clusters", len(clusters))

    self.set_params(**kwargs)
    self.setup_stats(clusters)


    # adj_graph is used to track adjacent partitions
    self.adj_graph = self.make_adjacency(clusters, self.partitions_complete)
    # rtree is static (computed once) to find base partitions within 
    # a merged partition
    self.rtree = self.construct_rtree(clusters)

    clusters_set = set(clusters)

    frontier, removed_clusters = get_frontier(clusters)
    _logger.debug("%d clusters in frontier", len(frontier))
    clusters_set.difference_update(removed_clusters)

    from matplotlib import pyplot as plt
    minv, maxv = 1e100, -1e100
    xs = np.arange(1000) / 1000.
    for c in clusters:
      ys = [c.inf_func(x) for x in xs]
      color = 'grey'
      alpha = 0.4
      minv = min(minv, min(ys))
      maxv = max(maxv, max(ys))
      plt.plot(xs, ys, alpha=alpha, color=color)


    while len(frontier) > 0:#self.min_clusters:
      print
      for c in sorted(clusters_set, key=lambda c: c.c_range[0]):
        print "%d\t%.4f\t%.4f" % (c.id, c.c_range[0], c.c_range[1])


      new_clusters, removed_clusters = self.expand_frontier(frontier)
      new_clusters.difference_update(frontier)
      new_clusters.difference_update(removed_clusters)


      # are these new clusters the best compared to _all_ past clusters?
      combined = set(new_clusters)
      combined.update(clusters_set)

      print "frontier of combined"
      for c in sorted(combined, key=lambda c: c.c_range[0]):
        print "%s\t%.4f\t%.4f" % (c, c.c_range[0], c.c_range[1])
  
      winners, losers = get_frontier(combined)
      if clusters_set.issuperset(winners):
        new_clusters = set()
      else:
        combined.difference_update(clusters_set)
        new_clusters = combined
        removed_clusters.update(losers)
        frontier = winners

      _logger.debug("%d new\t%d removed\t%d frontier\t%d clus_set", 
          len(new_clusters), len(removed_clusters), len(frontier), len(clusters_set))
      if not new_clusters: break

      map(self.adj_graph.remove, removed_clusters)
      map(self.adj_graph.insert, new_clusters)
      clusters_set.difference_update(removed_clusters)
      clusters_set.update(new_clusters)
      #frontier, removed_clusters = get_frontier(clusters_set)

    clusters_set = filter_bad_clusters(clusters_set)
    clusters_set = sorted(clusters_set, key=lambda c: c.error, reverse=True)


    for c in clusters_set:
      ys = [c.inf_func(x) for x in xs]
      color = 'red'
      alpha = .4
      minv = min(minv, min(ys))
      maxv = max(maxv, max(ys))
      plt.plot(xs, ys, alpha=alpha, color=color)
    plt.plot([self.c_range[0], self.c_range[0]], [minv, maxv], color='grey', alpha=0.4)
    plt.plot([self.c_range[1], self.c_range[1]], [minv, maxv], color='grey', alpha=0.4)
    plt.savefig("/Users/sirrice/infs.pdf")



    print "returning!"
    for c in sorted(clusters_set, key=lambda c: c.c_range[0]):
      print "\t%s\t%.4f, %.4f\t%.4f" % (c, c.c_range[0], c.c_range[1], c.error)
    return clusters_set
  


  def expand_frontier(self, frontier):
    """
    Return (newclusters, rmclusters)
    """

    newclusters, rmclusters = set(), set()
    for cluster in frontier:
      merges, rms = self.expand(cluster)
      merges = [m for m in merges if valid_number(m.error)]
      newclusters.update(merges)
      rmclusters.update(rms)
    return newclusters, rmclusters





  @instrument
  def dim_merge(self, cluster, dim, dec=None, inc=None):
    merged = Merger.dim_merge(self, cluster, dim, dec, inc)
    merged.c_range = list(self.c_range)
    merged.inf_func = merged.create_inf_func(self.learner.l)
    return merged


  @instrument
  def disc_merge(self, cluster, dim, vals):
    merged = Merger.disc_merge(self, cluster, dim, vals)
    merged.c_range = list(self.c_range)
    merged.inf_func = merged.create_inf_func(self.learner.l)
    return merged


  def expand(self, c):
    """
    Returns a frontier of clusters expanded from c that
    are possible optimals in _some_ c range

    XXX: optimization could be a minimum c range a cluster must be
          a candidate over
    """
    cur_bests = set([c])
    ret = set()
    seen = set()
    rms = set()
    while cur_bests:
      cur = cur_bests.pop()
      if cur in seen: 
        _logger.debug("seen \t%s", cur)
        ret.add(cur)
        continue
      seen.add(cur)
      expansions = self.expand_candidates(cur, seen)

      _logger.debug("expand\t%s", cur)
      
      dim_to_bests = defaultdict(set)
      for dim, direction, g in expansions:
        dim_bests = set([cur])

        for cand in g:
          cand_list = [cand]
          expanded = set()
          read = set()
          for o in dim_bests:
            new_cand_list = []
            read.add(o)
            for c in cand_list:
              left, right = intersect_c_ranges(c, o)
              new_cand_list.extend(left)
              expanded.update(right)
            cand_list = new_cand_list
            if not cand_list: break

          dim_bests.difference_update(read)
          dim_bests.update(expanded)
          dim_bests.update(cand_list)

        if cur in dim_bests: dim_bests.remove(cur)
        dim_to_bests[(dim, direction)] = dim_bests

      if True:
        allmerges = set()
        map(allmerges.update, dim_to_bests.values())
        merges = set(filter(lambda c: c.c_range[0] < c.c_range[1], allmerges))
        _logger.debug("%d all expanded clusters\t%d clusters", len(allmerges), len(merges))

        combined = set(cur_bests)
        combined.update(merges)

        frontier, losers = get_frontier(combined)
        if cur_bests.issuperset(frontier):
          new_bests = set()
        else:
          frontier.difference_update(cur_bests)
          cur_bests.difference_update(losers)
          new_bests = frontier

      else:
        # remove dominated clusters
        for outer_dim, oclusters in dim_to_bests.iteritems():
          for inner_dim, iclusters in dim_to_bests.iteritems():
            if outer_dim == inner_dim: break

            orms, irms = set(), set()

            for oc in oclusters:
              for ic in iclusters:
                if oc in orms: continue
                if ic in irms: continue
                newoc, newic = intersect_c_ranges(oc, ic)
                if newoc is None:
                  orms.add(oc)
                if newic is None:
                  irms.add(ic)
            oclusters.difference_update(orms)
            iclusters.difference_update(irms)

        new_bests = set()
        map(new_bests.update, dim_to_bests.values())

      if not new_bests: 
        ret.add(cur)
        continue

      for merged in new_bests:
        rms.update(merged.parents)
        self.adj_graph.insert(merged)
      seen.update(rms)

      cur_bests.update(new_bests)
      cur_bests, _ = get_frontier(cur_bests)
      _logger.debug("expand\t%d new bests\n\t%s", 
          len(new_bests), ",".join([str(c.id) for c in cur_bests]))

    return ret, rms


