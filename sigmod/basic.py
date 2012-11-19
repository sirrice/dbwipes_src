import bsddb3
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
from errfunc import ErrTypes


inf = 1e100000000000

class Basic(object):


    def __init__(self, **kwargs):
        self.aggerr = kwargs.get('aggerr', None)
        self.cols = list(self.aggerr.agg.cols)
        self.err_func = kwargs.get('err_func', self.aggerr.error_func.clone())
        self.merger = None
        self.params = {}
        self.costs = {}

        self.bad_thresh = 0
        self.good_thresh = 0
        self.min_pts = kwargs.get('min_pts', 5)
        self.max_bests = 20
        self.max_complexity = kwargs.get('max_complexity', 4)
        self.granularity = kwargs.get('granularity', 100)

        self.l = kwargs.get('l', 0.5)
        self.c = kwargs.get('c', 1.)
        self.bincremental = kwargs.get('bincremental', True)
        self.use_cache = kwargs.get('use_cache', False)

        self.tablename = kwargs.get('tablename', None)


        self.cache = bsddb3.hashopen('./dbwipes.cache')
        

        self.scorer_cost = 0.
        self.merge_cost = 0.

        self.set_params(**kwargs)

    def __hash__(self):
        components = [
                self.__class__.__name__,
                str(self.aggerr.__class__.__name__),
                str(set(self.cols)),
                self.err_func.__class__.__name__,
                self.tablename,
                self.l,
                self.c
                ]
        components = map(str, components)
        return hash('\n'.join(components))
                

    def set_params(self, **kwargs):
        self.cols = kwargs.get('cols', self.cols)
        self.use_cache = kwargs.get('use_cache', self.use_cache)
        self.params.update(kwargs)

    def setup_tables(self, full_table, bad_tables, good_tables, **kwargs):
        """
        create bad_err_funcs
        """
        self.full_table = full_table
        self.dummy_table = Orange.data.Table(full_table.domain)
        self.bad_tables = bad_tables
        self.good_tables = good_tables
        self.bad_err_funcs = [self.err_func.clone() for t in bad_tables]
        self.good_err_funcs = [self.err_func.clone() for t in good_tables]

        for ef, t in zip(self.bad_err_funcs, bad_tables):
            ef.setup(t)

        for ef, t in zip(self.good_err_funcs, good_tables):
            ef.errtype.errtype = ErrTypes.EQUALTO
            ef.setup(t)

        domain = self.full_table.domain
        attrnames = [attr.name for attr in domain]
        self.cont_dists = dict(zip(attrnames, Orange.statistics.basic.Domain(self.full_table)))
        self.disc_dists = dict(zip(attrnames, Orange.statistics.distribution.Domain(self.full_table)))



    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        @return final_clusters
        """

        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        pass


    def influence_cluster(self, cluster):
        rule = cluster.to_rule(self.dummy_table, self.cols, cont_dists=self.cont_dists, disc_dists=self.disc_dists)
        return Basic.influence(self, rule)


    def influences(self, rule, cs=[]):
        """
        compute influences for a list of c values
        """
        bdeltas, bcounts = self.bad_influences(rule)
        gdeltas, gcounts = self.good_influences(rule)
        gdeltas = map(abs, gdeltas)
        
        ret = []
        for c in cs:
            binfs = [bdelta/(bcount**c) for bdelta,bcount in zip(bdeltas, bcounts) if bcount]
            ginfs = [gdelta for gdelta,gcount in zip(gdeltas, gcounts) if gcount]
            
            binf = binfs and np.mean(binfs) or -1e10000000
            ginf = ginfs and max(ginfs) or 0
            inf = self.l * binf - (1. - self.l) * ginf

            ret.append(inf)
        return ret



    def influence(self, rule, c=None):
        bdeltas, bcounts = self.bad_influences(rule)
        gdeltas, gcounts = self.good_influences(rule)
        gdeltas = map(abs, gdeltas)
        
        if c is None:
            c = self.c
        binfs = [bdelta/(bcount**c) for bdelta,bcount in zip(bdeltas, bcounts) if bcount]
        ginfs = [gdelta for gdelta,gcount in zip(gdeltas, gcounts) if gcount]

        
        binf = binfs and np.mean(binfs) or -1e10000000
        ginf = ginfs and max(ginfs) or 0
        inf = self.l * binf - (1. - self.l) * ginf

        rule.quality = inf
        return inf


    def bad_influences(self, rule):
        return self.compute_stat(rule, self.bad_err_funcs, self.bad_tables)

    def good_influences(self, rule):
        return self.compute_stat(rule, self.good_err_funcs, self.good_tables)


    def compute_stat(self, rule, err_funcs, tables):
        datas = rule and map(rule.filter_table, tables) or tables
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

        block = (maxv - minv) / self.granularity
        ranges = [[minv + i*block, minv + (i+1)*block] for i in xrange(self.granularity)]
        ranges[0][0] = -inf
        ranges[-1][1] = inf
        return ranges


    def get_all_clauses(self, attr, max_card):
        class Ret(object):
            def __init__(self, attr, max_card, par):
                self.attr = attr
                self.max_card = max_card
                self.par = par

            def __iter__(self):
                attrobj = self.par.full_table.domain[self.attr]
                if attrobj.var_type == Orange.feature.Type.Discrete:
                    return self.par.all_discrete_clauses(self.attr, self.max_card)
                else:
                    return self.par.all_cont_clauses(self.attr)
        return Ret(attr, max_card, self)

            
    def all_discrete_clauses(self, attr, max_card=None):
        all_vals = self.col_to_clauses[attr]
        attrobj = self.full_table.domain[attr]
        idx = self.full_table.domain.index(attrobj)
        
        if max_card:
            for card in xrange(1, max_card+1):
                for vals in combinations(all_vals, card):
                    vals = [orange.Value(attrobj, value) for value in vals]
                    yield orange.ValueFilter_discrete(
                            position = idx,
                            values = vals)
        else:
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


