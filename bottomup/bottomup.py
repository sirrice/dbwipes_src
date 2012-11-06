import math
import pdb
import random
import numpy as np
import time
import orange
import Orange
import sys
sys.path.extend(['.', '..'])

from multiprocessing import Process, Pool
from collections import defaultdict
from scipy.spatial import KDTree
from rtree.index import Index as RTree
from rtree.index import Property as RProp
from operator import mul, and_

from util import rm_attr_from_domain, get_logger
from bounding_box import *
from cluster import *
from merger.merger import Merger
from merger.reexec import ReexecMerger
from zero import Zero
from learners.cn2sd.rule import SDRule


_logger = get_logger()

# compute sample needed to hit at least one of the error points
# compute R-nearest neighbors of samples using LSH
# iteratively merge clusters
# - sweep through continuous dimensions
# - merge based on cost function
#   - knob for size vs "purity"
# - how to deal with gaps?
# - whats the right radius (for RNN)




class BottomUp(object):
    def __init__(self, **kwargs):
        """
        """
        self.cols = []
        self.err_func = None
        self.perc_k = 0.01
        self.r = 5
        self.nsamples = None
        self.nstds = 1.
        self.merger = None
        self.is_mergable = None
        
        self.sample_cost = 0.
        self.kd_cost = 0.
        self.initclusters_cost = 0.
        self.merge_cost = 0.

        self.params = {}
        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        self.perc_k = kwargs.get('perc_k', self.perc_k)
        self.r = kwargs.get('r', self.r)
        self.nsamples = kwargs.get('nsamples', self.nsamples)
        self.cols = kwargs.get('cols', self.cols)
        self.err_func = kwargs.get('err_func', self.err_func)
        self.merger = kwargs.get('merger', self.merger)
        self.nstds = kwargs.get('nstds', self.nstds)
        self.is_mergable = kwargs.get('is_mergable', self.is_mergable)

        self.params.update(kwargs)


    def setup(self, table):
        self.table = table
        self.data = table.to_numpyMA('ac')[0].data

        self.all_cols = range(len(table.domain))
        self.pred_columns = [table.domain.index(table.domain[c]) for c in self.cols]
        self.search_cols = [pos for pos in self.pred_columns
                            if table.domain[pos].varType != orange.VarTypes.Discrete]
        self.search_names = [table.domain[pos].name for pos in self.search_cols]
        self.disc_cols = [pos for pos in self.pred_columns
                          if table.domain[pos].varType == orange.VarTypes.Discrete]
        self.disc_names = [table.domain[pos].name for pos in self.disc_cols]
        
        self.search_data = self.data[:,self.search_cols]
        self.search_bbox = points_bounding_box(self.search_data)
        self.disc_data = self.data[:,self.disc_cols]

        self.params.update({'cols' : tuple(self.cols),
                            'table': self.table,
                            'err_func' : self.err_func})
        

    def __call__(self, table):
        """
        base_table is the table used to setup the error function
        
        the table contains columns for:
        - executing the error function
        - columns that define the search space

        thus the kd and rtree indexes only index the columns
        in the search space.

        ## Columns Organization
        columns store the positions of the columns in the table numpy array
        e.g., [0, 2, 3] instead of ['voltage', 'humidity']
        
        We need to keep track of three different sets of columns:
        all columns: columns used to construct predicates + columns used by error function
        pred columns: columns used to construct predicates
        search columns: continuous value columns used for KNN
        disc columns: discrete value columns used to construct predicates

        search columns <= pred columns < all columns
        self.search_cols <= self.pred_cols < self.all_cols
        
        """
        self.setup(table)


        start = time.time()
        self.samples = self.construct_centroid_samples(self.get_sample_size())
        self.sample_cost = time.time() - start
        
        start = time.time()
        #self.kdtree = KDTree(self.samples)
        self.kdtree = KDTree(self.search_data)
        self.kd_cost = time.time() - start

        start = time.time()
        self.all_clusters = self.construct_initial_clusters()
        self.initclusters_cost = time.time() - start


        start = time.time()
        if not self.is_mergable:
            self.filterthreshold = compute_clusters_threshold(self.all_clusters, self.nstds)
            is_mergable = lambda c: c.error >= self.filterthreshold
        else:
            is_mergable = self.is_mergable

        self.merger = ReexecMerger(**self.params)
        self.final_clusters = self.merger(self.all_clusters, is_mergable=is_mergable)
        self.merge_cost = time.time() - start

        _logger.debug("returning %d clusters", len(self.final_clusters))

        return self.final_clusters



    def compute_filter_threshold(self, clusters, nstds=1.):
        errors = np.array([c.error for c in clusters])
        maxv, minv = errors.max(), errors.min()
        #if maxv > minv:
        #    errors = (errors - minv) / (maxv - minv)
        return min(max(errors), np.mean(errors) + nstds * np.std(errors))


    def get_sample_size(self):
        if self.nsamples:
            return min(len(self.data), self.nsamples)
        nsamples = int(1. / self.perc_k)
        if nsamples <= 0:
            return 1
        return min(len(self.data), nsamples)
        

    def construct_initial_clusters(self):
        npts = self.kdtree.data.shape[0]        
        samples = self.samples
        perc = (float(self.perc_k)) ** (1. / self.data.shape[1])
        search_bbox = self.search_bbox
        upper = sum(((x-n) * perc)**2 for n, x in zip(*search_bbox))
        upper = math.sqrt(upper)
        if len(samples) <= 1 or upper == 0:
            upper = 1e-10
        _logger.debug('upper bound\t%.4f', upper)
        _logger.debug("querying kdtree")
        k = max(2, int(npts * self.perc_k))

        # clusters = [self.cluster_from_rowids([n]) for n in xrange(npts)]
        # return clusters
        
        kdresult = self.kdtree.query(samples, k=k, distance_upper_bound=upper)
        kdresult = (idxs[idxs != npts] for idxs in kdresult[1])            
        clusters = []
        for idx, rowids in enumerate(kdresult):
            cluster = self.cluster_from_rowids(rowids)
            # k = self.k
            # nfound = len(rowids)
            # while cluster.volume == 0:
            #     k = max(k+5, int(k * 1.5))
            #     if k >= npts:
            #         break
            #     distances, indices = self.kdtree.query(samples[idx], k=k, distance_upper_bound=upper)
            #     rowids = indices[indices != npts]
            #     cluster = self.cluster_from_rowids(rowids)
            #     if len(rowids) == nfound:
            #         break
            #     nfound = len(rowids)
                
            if cluster:
                clusters.append(cluster)



        #distances, indices = self.kdtree.query(self.search_data, k=1)
        # clusters = []
        # for clusterid in xrange(npts):
        #     rowids = np.nonzero(indices == clusterid)
        #     cluster = self.cluster_from_rowids(rowids)
        #     if cluster:
        #         clusters.append(cluster)
        # return clusters
            


        # kdresult = self.kdtree.query_ball_point(
        #     samples,
        #     r=upper,
        #     eps=0.001
        #     )
        # clusters = filter(lambda c: c, map(self.cluster_from_rowids, kdresult))
        
        return clusters

    def cluster_from_rowids(self, rowids):
        err = self.err_func(self.data[rowids]) / len(rowids)
        bbox = points_bounding_box(self.search_data[rowids])
        disc = self.disc_data[rowids]

        if math.isnan(err):
            pdb.set_trace()
            self.err_func(self.data[rowids]) / len(rowids)

        discretes = {}
        for idx in xrange(disc.shape[1]):
            discretes[self.disc_names[idx]] = set(disc[:,idx])

        return Cluster(bbox, err, self.search_names, discretes=discretes, npts=len(rowids), rowids=rowids)


    def construct_centroid_samples(self, sample_size):
        """
        uses reservoir sampling
        http://en.wikipedia.org/wiki/Reservoir_sampling
        """
        data = self.data
        samples = range(min(len(data), sample_size))
                
        for idx in xrange(len(samples), self.search_data.shape[0]):
            i = random.randint(0, idx)
            if i < sample_size:
                samples[i] = idx

        samples = list(set(samples))
        samples = self.search_data[samples,:]
        _logger.debug("returning %d samples", sample_size)
        return samples


class ZeroBottomUp(BottomUp):

    def __init__(self, *args, **kwargs):
        self.zero = None
        BottomUp.__init__(self, *args, **kwargs)
        

    def setup(self, table):
        """
        normalize the data ranges before calling parent setup with normalized table
        """

        self.data = table.to_numpyMA('ac')[0]
        
        pred_columns = [table.domain.index(table.domain[c]) for c in self.cols]
        search_cols = [pos for pos in pred_columns
                       if table.domain[pos].varType != orange.VarTypes.Discrete]

        bounds = Zero.compute_bounds(self.data[:, search_cols])
        self.zero = Zero(search_cols, bounds=bounds)
        self.zerod_data = self.zero.zero(self.data)
        self.zerod_table = Orange.data.Table(table.domain, self.zerod_data.data)
        super(ZeroBottomUp, self).setup(self.zerod_table)

        
        

    def __call__(self, table, **kwargs):

        BottomUp.__call__(self, table, **kwargs)

        # unzero all clusters and final clusters
        self.all_clusters = map(self.zero.unzero_cluster, self.all_clusters)
        self.final_clusters = map(self.zero.unzero_cluster, self.final_clusters)

        return self.final_clusters




class DiscreteBottomUpF(object):
    def __init__(self, params):
        self.params = dict(params)
    
    def __call__(self, args):
        partition_keys, cols, continuous_table = args

        self.params['cols'] = cols
        self.params['is_mergable'] = lambda c: True

        
        bottomup = ZeroBottomUp(**self.params)
        clusters = bottomup(continuous_table)
        clusters = bottomup.all_clusters
        
        for c in clusters:
            for name, keys in partition_keys.iteritems():
                c.discretes[name].update(keys)

        stats = (bottomup.kd_cost,
                 bottomup.sample_cost,
                 bottomup.initclusters_cost,
                 bottomup.merge_cost)
        return stats, clusters


class DiscreteBottomUp(ZeroBottomUp):

    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.parallelize = True
        self.nprocesses = 4
        
        super(DiscreteBottomUp, self).__init__(**kwargs)


    def set_params(self, **kwargs):
        super(DiscreteBottomUp, self).set_params(**kwargs)
        self.parallelize = kwargs.get('parallelize', self.parallelize)
        self.nprocesses = kwargs.get('nprocesses', self.nprocesses)        

    def get_partition(self, bits, discrete_vals, table):
        domain = table.domain
        conds = []
        for bit, (name, vals) in zip(bits, discrete_vals):
            attr = domain[name]
            pos = domain.index(attr)
            conds.append(orange.ValueFilter_discrete(
                position=pos,
                values=[orange.Value(attr, vals[bit])]))
        f = orange.Filter_values(domain=table.domain,
                                 conditions=conds)

        partition = f(table)
        return partition


    def discrete_vals_iter(self, table):
        """
        @param
        @return tuple of position bits, parameters for calling bottomup, and the vertical subset of the table
                position bits is actually a list of indexes that will be added to the resulting clusters
        """
        ddists = Orange.statistics.distribution.Domain(table)
        discrete_vals = []
        for pos, attr in enumerate(table.domain):
            if attr.var_type == Orange.feature.Type.Discrete:
                discrete_vals.append((attr.name, attr.values))

        cards = [len(vals) for name, vals in discrete_vals]
        if not cards:
            return

        
        all_clusters = []
        for bits in bitarray_iterator(cards):
            partition = self.get_partition(bits, discrete_vals, table)
            if not len(partition):
                continue

            
            rmcols = [attr.name for attr in partition.domain
                      if attr.var_type == Orange.feature.Type.Discrete and attr.name in self.cols]
            cols = [col for col in self.cols if col not in rmcols]
            continuous_table = rm_attr_from_domain(partition, rmcols)

            partition_keys = {}
            for bit, (name, vals) in zip(bits, discrete_vals):
                partition_keys[name] = [bit]
            
            yield partition_keys, cols, continuous_table
                    


    def __call__(self, table, **kwargs):
        """
        partition table by the discrete attributes
        """
        
        # setup the error function to execute on slim table
        rmcols = [attr.name for attr in table.domain
                  if attr.var_type == Orange.feature.Type.Discrete and attr.name in self.cols]
        thin_table = rm_attr_from_domain(table, rmcols)
        self.params['err_func'].setup(thin_table)
        bottomup_func = DiscreteBottomUpF(self.params)        
        
        if self.parallelize:
            start = time.time()
            pool = Pool(self.nprocesses)
            results = pool.map(bottomup_func, self.discrete_vals_iter(table))
            pool.close()
            self.merge_cost += time.time() - start
        else:
            results = []
            for args in self.discrete_vals_iter(table):
                results.append(bottomup_func(args))

            if results:
                all_stats, clusters_list = zip(*results)                
                kd_cost, sample_cost, initclusters_cost, merge_cost = zip(*all_stats)
                self.kd_cost += sum(kd_cost)
                self.sample_cost += sum(sample_cost)
                self.initclusters_cost += sum(initclusters_cost)
                self.merge_cost += sum(merge_cost)

        
        if not len(results):
            return super(DiscreteBottomUp, self).__call__(table)

        all_stats, clusters_list = zip(*results)
        self.all_clusters = []
        map(self.all_clusters.extend, clusters_list)

        BottomUp.setup(self, table)
        thresh = compute_clusters_threshold(self.all_clusters)
        final_clusters = self.normalize_results(self.all_clusters,
                                                is_mergable=lambda c: c.error >= thresh)

        self.final_clusters = final_clusters
        
        return final_clusters


    def normalize_results(self, all_clusters, **kwargs):
        """
        take clusters generated in parallel and merge/normalize them
        """
        if not all_clusters:
            return all_clusters
        
        start = time.time()
        params = dict(self.params)
        params.update(kwargs)
        params.update({'is_mergable' : kwargs.get('is_mergable', lambda c: True),
                       'cols' : tuple(self.cols),
                       'table': self.table,
                       'err_func' : self.err_func})
        merger = ReexecMerger(**params)
        ret = merger(all_clusters)
        self.merge_cost += time.time() - start

        return ret


            

    
        
def bitarray_iterator(cards):
    bits = np.zeros(len(cards)).astype(int)
    maxval = reduce(mul, cards)
    while True:
        idx = len(bits) - 1
        while idx >= 0:
            bits[idx] += 1
            if bits[idx] >= cards[idx]:
                bits[idx] = 0
                idx -= 1
            else:
                break
        if idx < 0:
            return 
        yield bits


