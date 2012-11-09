import math
import pdb
import random
import numpy as np
import sys
import time
sys.path.extend(['.', '..'])

from itertools import chain
from collections import defaultdict
from rtree.index import Index as RTree
from rtree.index import Property as RProp
from operator import mul, and_, or_

from util import rm_attr_from_domain, get_logger
from util.table import *
from bottomup.bounding_box import *
from bottomup.cluster import *
from zero import Zero

_logger = get_logger()

class AdjacencyGraph(object):
    def __init__(self, clusters):
        self.graph = defaultdict(set)
        self.cid = 0
        self.clusters = []
        self.id2c = dict()
        self.c2id = dict()

        map(self.insert, clusters)

    def insert(self, cluster):
        if cluster in self.graph:
            return

        self.graph[cluster] = set()

        for o in self.graph.keys():
            if cluster != o and cluster.adjacent(o):
                self.graph[cluster].add(o)
                self.graph[o].add(cluster)
        

        cid = len(self.clusters)
        self.clusters.append(cluster)
        self.id2c[cid] = cluster
        self.c2id[cluster] = cid


    def remove(self, cluster):
        if cluster not in self.graph:
            return

        for neigh in self.graph[cluster]:
            self.graph[neigh].remove(cluster)
        del self.graph[cluster]

        cid = self.c2id[cluster]
        del self.c2id[cluster]
        del self.id2c[cid]
        self.clusters[cid] = None

    def neighbors(self, cluster):
        if cluster in self.graph:
            return self.graph[cluster]
        ret = set()
        for key in filter(cluster.adjacent, self.graph.keys()):
            ret.update(self.graph[key])
        return ret



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
        
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        self.min_clusters = kwargs.get('min_clusters', self.min_clusters)
        self.is_mergable = kwargs.get('is_mergable', self.is_mergable)
        # lambda cluster: influence_value_of(cluster)
        self.influence = kwargs.get('influence', self.influence)

        

    def setup_stats(self, clusters):
        """
        computes error bounds and the minimum volume of a 0-volume cluster

        """
        vols = np.array([c.volume for c in clusters if c.volume > 0])
        if len(vols) == 0:
            self.point_volume = 0.00001
        else:
            self.point_volume = vols.min() / 2.

        #self.setup_errors(clusters)

#        # setup table bounds to that calls to kdtree and rtree are zeroed
#        # this should be transparent from the caller
#        self.cont_cols = continuous_columns(self.table, self.cols)
#        self.cont_pos = column_positions(self.table, self.cont_cols)
#        self.search_data = self.table.to_numpyMA('ac')[0].data[:, self.cont_pos]
#        self.search_bbox = points_bounding_box(self.search_data)
#
#        self.zero = Zero(range(len(self.cont_pos)), bounds=self.search_bbox)

    def setup_errors(self, clusters):
        return
        errors = np.array([c.error for c in clusters])
        self.mean_error = np.mean(errors)
        pdb.set_trace()
        self.std_error = np.std(errors)
        self.min_error = errors.min()
        self.max_error = errors.max()
        self.diff_error = (self.max_error - self.min_error) or 1.
        

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

    def get_intersection(self, rtree, bbox):
        if len(bbox[0]) == 0:
            return rtree.intersection([0, 0, 1, 1])

        #bbox = self.scale_box(bbox)
        if len(bbox[0]) == 1:
            return rtree.intersection(bbox[0] + (0,) + bbox[1] + (1,))
        return rtree.intersection(bbox[0] + bbox[1])        


    def merge(self, cluster, neighbor, clusters, rtree):
        newbbox = bounding_box(cluster.bbox, neighbor.bbox)
        cidxs = self.get_intersection(rtree, newbbox)
        intersecting_clusters = [clusters[cidx] for cidx in cidxs]

        merged = Cluster.merge(cluster, neighbor, intersecting_clusters, self.point_volume)
        if not merged:
            pdb.set_trace()
            return None

        intersecting_clusters = filter(merged.contains, intersecting_clusters)
        merged.idxs = set(chain(cluster.idxs, neighbor.idxs, *[ic.idxs for ic in intersecting_clusters]))
        merged.error = self.influence(merged)
        return merged

    def setup_graph(self, clusters):
        matrix = np.zeros((len(clusters), len(clusters)))
        for i1 in xrange(len(clusters)):
            clusters[i1].idxs = [i1]
            for i2 in xrange(i1+1, len(clusters)):
                if clusters[i1].adjacent(clusters[i2]):
                    matrix[i1,i2] = matrix[i2,i1] = 1
        return matrix


    def neighbors(self, idxs, matrix):
        ret = set()
        nonzeros = map(np.nonzero, matrix[tuple(idxs),:] == 1)
        map(ret.update, (arr[0].tolist() for arr in nonzeros))
        ret.difference_update(idxs)
        return ret


    def expand(self, cluster, clusters, adj_graph, rtree):
        rms = set()
        while True:
            neighbors = adj_graph.neighbors(cluster)
            f = lambda n: self.merge(cluster, n, clusters, rtree)
            
            rms, tomerges = set(), list()
            for n in neighbors:
                if cluster.contains(n) or cluster.same(n, epsilon=0.05):
                    rms.add(n)
                else:
                    tomerges.append(n)

            tomerges.sort(key=lambda n: n.error, reverse=True)
            merged = None
            for tomerge in tomerges:
                merged = f(tomerge)
                if merged.error == -1e100000:
                    continue
                if merged.error <= max(cluster.error, tomerge.error):
                    continue
                break

            if not merged:
                break

            _logger.debug('\t%s',merged)

            rms.update(merged.parents)
            cluster = merged
        _logger.debug('\n')
        return cluster, rms



        
    def __call__(self, clusters, **kwargs):
        if not clusters:
            return list(clusters)

        _logger.debug("merging %d clusters", len(clusters))

        self.set_params(**kwargs)
        self.setup_stats(clusters)
        clusters_set = set(clusters)
        adj_matrix = AdjacencyGraph(filter(self.is_mergable, clusters))


        results = []



        while len(clusters_set) > self.min_clusters:
            self.setup_errors(clusters_set)

            cur_clusters = sorted(clusters_set, key=lambda c: c.error, reverse=True)
            mergable_clusters = filter(self.is_mergable, cur_clusters)
            mergable_clusters.sort(key=lambda mc: mc.error, reverse=True)
            rtree = self.construct_rtree(cur_clusters)


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
                        _logger.debug("skipped\n\t%s\n\t%s", str(cluster), str(test))
                        skip = True
                        break
                if skip:
                    continue

                merged, rms = self.expand(cluster, cur_clusters, adj_matrix, rtree) 
                if not merged or merged == cluster or len(filter(lambda c: c.contains(merged), cur_clusters)):
                    seen.add(cluster)
                    continue
                
                _logger.debug("%.4f\t%.4f\t-> %.4f",
                               merged.parents[0].error,
                               merged.parents[1].error,
                               merged.error)
                seen.update(merged.parents)


                if merged not in cur_clusters:
                    new_clusters.add(merged)                    
                merged_clusters.update(rms)

            _logger.debug("merged %d\t%d new clusters\tout of %d",
                          len(merged_clusters),
                          len(new_clusters),
                          len(mergable_clusters))

            
            if not new_clusters:
                break


            map(adj_matrix.remove, merged_clusters)
            map(adj_matrix.insert, new_clusters)
            clusters_set.difference_update(merged_clusters)
            clusters_set.update(new_clusters)


        return sorted(clusters_set, key=lambda c: c.error, reverse=True)
            
    

