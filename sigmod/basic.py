import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from itertools import chain


from learners.cn2sd.rule import fill_in_rules
from learners.cn2sd.refiner import *
from score import QuadScoreSample7
from bottomup.bounding_box import *
from bottomup.cluster import *
from util import *




class Basic(object):


    def __init__(self, **kwargs):
        self.aggerr = kwargs.get('aggerr', None)
        self.cols = list(self.aggerr.agg.cols)
        self.err_func = kwargs.get('err_func', self.aggerr.error_func.clone())
        self.merger = None
        self.params = {}

        self.bad_thresh = 0
        self.good_thresh = 0
        self.min_pts = 5
        self.max_bests = 100

        self.lamb = kwargs.get('lamb', 0.5)
        self.c = kwargs.get('c', 1)
        self.bincremental = kwargs.get('bincremental', True)
        

        self.scorer_cost = 0.
        self.merge_cost = 0.

        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        self.cols = kwargs.get('cols', self.cols)
        self.params.update(kwargs)

    def setup_tables(self, full_table, bad_tables, good_tables, **kwargs):
        """
        create bad_err_funcs
        """
        self.full_table = full_table
        self.bad_tables = bad_tables
        self.good_tables = good_tables
        self.bad_err_funcs = [self.err_func.clone() for t in bad_tables]
        self.good_err_funcs = [self.err_func.clone() for t in good_tables]

        for ef, t in zip(chain(self.bad_err_funcs, self.good_err_funcs),
                         chain(bad_tables, good_tables)):
            ef.setup(t)


    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        @return final_clusters
        """

        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        pass

    def influence(self, rule):
        bdeltas, bcounts = self.bad_influences(rule)
        gdeltas, gcounts = self.good_influences(rule)

        binfs = [bdelta/(bcount**self.c) for bdelta,bcount in zip(bdeltas, bcounts)]
        ginfs = [gdelta/(gcount**self.c) for gdelta,gcount in zip(gdeltas, gcounts)]
        
        binf = np.mean(binfs)
        ginf = max(ginfs)
        inf = self.lamb * binf - (1. - self.lamb) * ginf

        rule.quality = inf
        return inf


    def bad_influences(self, rule):
        return self.compute_stat(rule, self.bad_err_funcs, self.bad_tables)

    def good_influences(self, rule):
        return self.compute_stat(rule, self.good_err_funcs, self.good_tables)


    def compute_stat(self, rule, err_funcs, tables):
        datas = map(rule.filter_table, tables)
        infs = []
        for ef, data in zip(err_funcs, datas):
            arr = data.to_numpyMA('ac')[0]
            inf = ef(arr.data)
            infs.append(inf)
        return infs, map(len, datas)






    def all_unit_clauses(self, attr):
        # continuous: split 1000 ways, uniformly
        # discrete: every unique value
        attrobj = self.full_table.domain[attr]
        idx = self.full_table.domain.index(attrobj)
        if attrobj.var_type == Orange.feature.Type.Discrete:
            ddist = Orange.statistics.distribution.Domain(self.full_table)[idx]
            return ddist.keys()
        
        bdist = Orange.statistics.basic.Domain(self.full_table)[idx]
        minv, maxv = bdist.min, bdist.max
        if minv == maxv:
            return [[-inf, inf]]

        block = (maxv - minv) / 10.
        ranges = [[minv + i*block, minv + (i+1)*block] for i in xrange(10)]
        ranges[0][0] = -inf
        ranges[-1][1] = inf
        return ranges


    def get_all_clauses(self, attr):
        attrobj = self.full_table.domain[attr]
        if attrobj.var_type == Orange.feature.Type.Discrete:
            return self.all_discrete_clauses(attr)
        return self.all_cont_clauses(attr)

    def all_discrete_clauses(self, attr):
        all_vals = self.col_to_clauses[attr]
        attrobj = self.full_table.domain[attr]
        idx = self.full_table.domain.index(attrobj)

        for vals in powerset(all_vals):
            vals = [orange.Value(attrobj, value) for value in vals]
            yield orange.ValueFilter_discrete(
                    position = idx,
                    values = vals)

    def all_cont_clauses(self, attr):
        units = self.col_to_clauses[attr]
        idx = self.full_table.domain.index(self.full_table.domain[attr])
        for sidx in xrange(0, len(units)):
            for eidx in xrange(sidx, len(units)):
                minv = units[sidx][0]
                maxv = units[eidx][1]
                yield Orange.data.filter.ValueFilterContinuous(
                        position=idx,
                        oper=orange.ValueFilter.Between,
                        min=minv,
                        max=maxv)


