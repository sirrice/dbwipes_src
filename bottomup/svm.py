import pdb
import sys
import sklearn
import random
import numpy as np
sys.path.append('..')

from multiprocessing import Process, Pool
from sklearn import svm
from itertools import chain, groupby


from hybrid.hybrid import *
from bottomup import *


class SVMBottomUpF(object):
    def __init__(self, params):
        self.params = dict(params)
    
    def __call__(self, args):
        partition_keys, cols, continuous_table = args

        self.params['cols'] = cols
        self.params['is_mergable'] = lambda c: True
        self.params['k'] = max(1, int(self.params['perc_k'] * len(continuous_table)))

        
        bottomup = ZeroBottomUp(**self.params)
        bottomup.setup(continuous_table)
        bottomup.samples = bottomup.construct_centroid_samples(bottomup.get_sample_size())
        bottomup.kdtree = KDTree(bottomup.search_data)
        clusters = bottomup.construct_initial_clusters()
        clusters = map(bottomup.zero.unzero_cluster, clusters)
        
        for c in clusters:
            for name, keys in partition_keys.iteritems():
                c.discretes[name].update(keys)

        stats = (bottomup.kd_cost,
                 bottomup.sample_cost,
                 bottomup.initclusters_cost,
                 bottomup.merge_cost)
        return stats, clusters



class BUFoo(HybridBottomUp):
    def __init__(self, **kwargs):
        kwargs['err_func'] = kwargs.get('err_func', kwargs['aggerr'].error_func)
        HybridBottomUp.__init__(self, **kwargs)
        self.params = kwargs

    def __call__(self, table, rules, perc_k=0.001):
        if len(rules) == 0:
            self.all_clusters = []
            return []
        
        self.rules = rules
        
        # setup the error function to execute on slim table
        rmcols = [attr.name for attr in table.domain
                  if attr.var_type == Orange.feature.Type.Discrete and attr.name in self.cols]
        thin_table = rm_attr_from_domain(table, rmcols)
        self.params['err_func'].setup(thin_table)
        self.params['perc_k'] = perc_k
        bottomup_func = SVMBottomUpF(self.params)



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
        
        all_stats, clusters_list = zip(*results)
        self.all_clusters = []
        map(self.all_clusters.extend, clusters_list)


        # # consider original rules as clusters as well
        # self.params['err_func'].setup(table)
        # for rule in rules:
        #     error = self.params['err_func'](rule.examples.to_numpyMA('ac')[0].data)
        #     c = Cluster.from_rule(rule, self.cols, error=error)
        #     self.all_clusters.append(c)

        
        return self.all_clusters
        




"""
1) run top down, get discrete partitions
2) run zero on partitions and collect all clusters and final clusters
3) use all clusters and logregr to filter dimensions
4) remove extraneous dimenions from final clusters
5) merge
6) done
"""
class SVMBottomUp(Hybrid):
    def __init__(self, *args, **kwargs):
        Hybrid.__init__(self, *args, **kwargs)
        self.cols = kwargs['cols']
        self.bottomup = BUFoo(*args, **kwargs)



    def get_topdown_clusters(self, table, **kwargs):
        
        rules = self.topdown(table, **kwargs)
        all_rules = self.topdown.rules
        qs = [r.quality for r in all_rules]
        if len(all_rules) > 1:
            threshold = min(max(qs), np.mean(qs) + np.std(qs))
            good_rules = filter(lambda r: r.quality >= threshold, all_rules)
            threshold = max(min(qs), np.mean(qs))
            bad_rules = filter(lambda r: r.quality <= threshold, all_rules)
        else:
            good_rules = all_rules
            bad_rules = []


        good_clusters = self.bottomup(table, good_rules, perc_k=0.001)
        bad_clusters = self.bottomup(table, bad_rules, perc_k=0.05)

        # self.bottomup(table, all_rules, **kwargs)
        # self.all_clusters = self.bottomup.all_clusters
        # errors = [c.error for c in self.all_clusters]
        # threshold = np.mean(errors) + np.std(errors)
        # good_clusters = filter(lambda c: c.error >= threshold, self.all_clusters)
        # bad_clusters = filter(lambda c: c.error < threshold, self.all_clusters)
        return good_clusters, bad_clusters



    def project_clusters(self, good_clusters, bad_clusters):
        # # dimension reduction 
        fr = FeatureReducer()
        filtered_cols = set(fr(good_clusters, bad_clusters, self.cols))
        
        good_proj_clusters = [c.project(filtered_cols) for c in good_clusters]
        bad_proj_clusters = [c.project(filtered_cols) for c in bad_clusters]
        all_clusters = [good_proj_clusters, bad_proj_clusters]

        ret = []
        for clusters in all_clusters:
            cluster_groups = groupby(clusters, key=lambda c: c.bound_hash)
            projected_clusters = []
            for k, group in cluster_groups:
                group = list(group)
                c = group[0].clone()
                c.error = np.mean([gc.error for gc in group])
                projected_clusters.append(c)
            ret.append(projected_clusters)
        good_clusters, bad_clusters = tuple(ret)
        
        return filtered_cols, good_clusters, bad_clusters


        
    def __call__(self, table, **kwargs):
        start = time.time()

        good_clusters, bad_clusters = self.get_topdown_clusters(table, **kwargs)

        BottomUp.setup(self.bottomup, table)
        if good_clusters:
            filtered_cols, good_clusters, bad_clusters = self.project_clusters(good_clusters, bad_clusters)
            self.all_clusters = good_clusters + bad_clusters

            
            if len(good_clusters[0].centroid) == 0:
                self.final_clusters = good_clusters
            else:

                # final merging pass
                if len(bad_clusters) == 0:
                    errors = [c.error for c in good_clusters]
                    thresh = np.mean(errors) + np.std(errors)
                    projected_clusters = good_clusters
                else:
                    thresh = min(good_clusters, key=lambda c: c.error).error
                    projected_clusters = good_clusters + [min(bad_clusters, key=lambda c:c.error)]

                is_mergable = lambda c: c.error >= thresh

                print "%d mergable clusters" % len(filter(is_mergable, projected_clusters))

                self.bottomup.cols = filtered_cols

                kwargs = dict(self.params)
                kwargs['is_mergable'] = is_mergable
                self.final_clusters = self.bottomup.normalize_results(
                    projected_clusters,
                    **kwargs)

                self.final_clusters.sort(key=lambda c: c.error, reverse=True)
        else:
            self.all_clusters = good_clusters + bad_clusters
            self.final_clusters = self.all_clusters
                
        print "%d final clusters" % (len(self.final_clusters))


        self.cost = time.time() - start
        return self.final_clusters

        



class FeatureReducer(object):

    def discretes_iterator(self, discretes, keys):
        if not len(keys):
            yield []
            return
        
        all_vals = [list(discretes.get(key, None) or (-1,)) for key in keys]
        idxs = np.zeros(len(keys)).astype(int)

        while True:
            idx = len(idxs) - 1
            while idx >= 0:
                idxs[idx] += 1
                if idxs[idx] >= len(all_vals[idx]):
                    idxs[idx] = 0
                    idx -= 1
                else:
                    break
            if idx < 0:
                return
            yield [vals[idx] for idx, vals in zip(idxs, all_vals)]

    def __call__(self, good_clusters, bad_clusters, cols):
        """
        assumes that clusters contain all discrete values (even if empty)
        uses Logistic Regression w/ L1 regularization to kill number of dimensions
        """
        print "feature reduction on %d good clusters, %d bad clusters" % (len(good_clusters), len(bad_clusters))

        discretekeys = set()
        for c in chain(good_clusters):#, bad_clusters):
            discretekeys.update(c.discretes.keys())
        discretekeys = list(discretekeys)
        all_cols = cols + discretekeys

        sample_size = len(bad_clusters)
        if sample_size > 1000:
            sample_size = 1000
            bad_clusters = random.sample(bad_clusters, sample_size)

        Xs, Ys = [], []
        for c in chain(good_clusters, bad_clusters):
            
            for discrete_vals in self.discretes_iterator(c.discretes, discretekeys):
                for row in [list(c.centroid), list(c.bbox[0]), list(c.bbox[1])]:
                    Xs.append(row + discrete_vals)
                    Ys.append(c.error)

        # construct learner inputs and normalize them    
        Xs = np.array(Xs)
        Ys = np.array(Ys)
        zero = Zero(range(Xs.shape[1]), bounds=Zero.compute_bounds(Xs))
        Xs = zero.zero(Xs)
        # Ys = np.zeros(len(Xs))        
        # Ys[:len(good_clusters)] = 1
        # Ys[len(good_clusters):] = 0

        print "running logistic regression on %d points" % (len(Xs))
        learner = sklearn.linear_model.LogisticRegression(penalty='l1', fit_intercept=False)
        learner.fit(Xs, Ys)


        # filter on non-zero coefficients
        coefs = list(learner.coef_[0])
        print zip(all_cols, coefs)
        filtered_cols = [col for col, coef in zip(all_cols, coefs) if coef != 0]
        print filtered_cols

        return filtered_cols

        
            
