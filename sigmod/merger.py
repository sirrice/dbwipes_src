import math
import pdb
import random
import numpy as np
import sys
import time
sys.path.extend(['.', '..'])

from itertools import chain
from collections import defaultdict
from scipy.spatial import KDTree
from rtree.index import Index as RTree
from rtree.index import Property as RProp
from operator import mul, and_

from util import rm_attr_from_domain, get_logger
from util.table import *
from bottomup.bounding_box import *
from bottomup.cluster import *
from zero import Zero

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

        self.setup_errors(clusters)

#        # setup table bounds to that calls to kdtree and rtree are zeroed
#        # this should be transparent from the caller
#        self.cont_cols = continuous_columns(self.table, self.cols)
#        self.cont_pos = column_positions(self.table, self.cont_cols)
#        self.search_data = self.table.to_numpyMA('ac')[0].data[:, self.cont_pos]
#        self.search_bbox = points_bounding_box(self.search_data)
#
#        self.zero = Zero(range(len(self.cont_pos)), bounds=self.search_bbox)

    def setup_errors(self, clusters):
        errors = np.array([c.error for c in clusters])
        self.mean_error = np.mean(errors)
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

    def normalize_cluster_errors(self, clusters):
        if not self.diff_error:
            return

        for c in clusters:
            c.error = (c.error - self.min_error) / self.diff_error
        return clusters

    def unnormalize_cluster_errors(self, clusters):
        if not self.diff_error:
            return

        for c in clusters:
            c.error = c.error * self.diff_error + self.min_error

    def filter_discrete(self, c1, c2, intersecting_clusters):
        return filter(c1.discretes_intersect,
                      filter(c2.discretes_intersect, intersecting_clusters))

    def construct_rtree(self, clusters):
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


    def expand(self, cluster, clusters, adjacency_matrix, id_to_cluster, rtree):
        rms = set()
        while True:

            nidxs = self.neighbors(cluster.idxs, adjacency_matrix)
            neighbors = set(chain(*filter(lambda x:x, map(id_to_cluster.get, nidxs))))
            f = lambda n: self.merge(cluster, n, clusters, rtree)
            groups = dict(groupby(neighbors, cluster.contains))
            rms.update(groups.get(True, []))
            neighbors = groups.get(False, [])

            mergeds = sorted(map(f, neighbors), key=lambda n: n.error, reverse=True)

            if not mergeds or mergeds[0] == cluster:
                break

            merged = mergeds[0]
            if merged.error <= max(merged.parents[0].error, merged.parents[1].error):
                break
            print '\t',merged

            rms.update(merged.parents)
            cluster = merged
        return cluster, rms



        
    def __call__(self, clusters, **kwargs):
        if not clusters:
            return list(clusters)

        _logger.debug("merging %d clusters", len(clusters))

        self.set_params(**kwargs)
        self.setup_stats(clusters)
        clusters_set = set(clusters)
        adjacency_matrix = self.setup_graph(clusters)

        # horrible hack to retrieve a merged cluster from the 
        # contained cluster -- because adjacency matrix is computed once
        id_to_cluster = defaultdict(set)
        for mc in clusters:
            for idx in mc.idxs:
                id_to_cluster[idx].add(mc)

        results = []

#        while len(clusters_set) > self.min_clusters:
#
#            cur_clusters = sorted(clusters_set, key=lambda c: c.error, reverse=True)
#            all_rms = set()
#            all_merged = set()
#            for cluster in filter(self.is_mergable, cur_clusters):
#                # grow as much as possible
#                rtree = self.construct_rtree(cur_clusters)
#                merged, rms = self.expand(cluster, cur_clusters, adjacency_matrix, id_to_cluster, rtree)
#
#                all_rms.update(rms)
#                all_merged.add(merged)
#
#            for rm in all_rms:
#                for idx in rm.idxs:
#                    if rm in id_to_cluster[idx]:
#                        id_to_cluster[idx].remove(rm)
#            for merged in all_merged:
#                for idx in merged.idxs:
#                    id_to_cluster[idx].add(merged)
#
#            _logger.debug("#merged\t%d\tout of %d", len(all_merged), len(cur_clusters))
#
#            if not merged:
#                break
#
#            clusters_set.difference_update(rms)
#            clusters_set.update(all_merged)
#
#        return results
#




        while len(clusters_set) > self.min_clusters:
            self.setup_errors(clusters_set)

            cur_clusters = sorted(clusters_set, key=lambda c: c.error, reverse=True)
            mergable_clusters = filter(self.is_mergable, cur_clusters)
            mergable_clusters.sort(key=lambda mc: mc.error, reverse=True)
            rtree = self.construct_rtree(cur_clusters)
            for mc in cur_clusters:
                for idx in mc.idxs:
                    id_to_cluster[idx].add(mc)


            _logger.debug("# mergable clusters: %d\tout of\t%d",
                    len(mergable_clusters),
                    len(cur_clusters))
            if not mergable_clusters:
                break



            merged_clusters, new_clusters = set(), set()
            seen = set()

            for cluster in mergable_clusters:
                if cluster in merged_clusters or cluster in new_clusters or cluster in seen:
                    continue

                merged, rms = self.expand(cluster, cur_clusters, adjacency_matrix, id_to_cluster, rtree) 
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


            clusters_set.difference_update(merged_clusters)
            clusters_set.update(new_clusters)
            for mc in merged_clusters:
                for idx in mc.idxs:
                    if mc in id_to_cluster[idx]:
                        id_to_cluster[idx].remove(mc)
                        if not id_to_cluster[idx]:
                            del id_to_cluster[idx]


        #self.unnormalize_cluster_errors(clusters_set)
        return sorted(clusters_set, key=lambda c: c.error, reverse=True)
            
    

