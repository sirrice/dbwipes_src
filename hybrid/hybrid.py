import time
import pdb
import numpy as np
import sys
sys.path.extend(['.', '..'])


from util import rm_attr_from_domain, get_logger, wmean, wstd
from score import QuadScoreSample7
from bottomup.bounding_box import *
from bottomup.cluster import *
from bottomup.bottomup import DiscreteBottomUp
from topdown.topdown import TopDown



class HybridTopDown(TopDown):
    def __call__(self, table, **kwargs):
        """
        table has been trimmed of extraneous columns.
        Topdown will use all columns to construct rules
        """
        self.set_params(**kwargs)
        
        self.err_func.setup(table)
        self.table = table

        self.col_positions = [table.domain.index(table.domain[c]) for c in self.cols]
        self.disc_positions = [pos for pos in self.col_positions
                               if table.domain[pos].varType == orange.VarTypes.Discrete]
        self.disc_cols = [table.domain[pos] for pos in
                          self.disc_positions]

        self.rules = list(self.get_scorer_rules(table,
                                           self.disc_cols,
                                           self.err_func,
                                           **kwargs))

        self.fill_in_rules(self.rules, table)

        self.top_rules = list(self.filter_top_rules(self.rules))
        return self.top_rules


    def filter_top_rules(self, rules, nstds=1.):
        if len(rules) <= 1:
            return rules

        errors = [r.quality for r in rules]
        npts = [len(r.examples) for r in rules]
        maxv, mean, std = max(errors), wmean(errors, npts),wstd(errors, npts)
        thresh = min(maxv, mean + nstds * std)
        f = lambda c: c.quality >= thresh
        return filter(f, rules)


class HybridBottomUp(DiscreteBottomUp):
    def __init__(self, **kwargs):
        DiscreteBottomUp.__init__(self, **kwargs)

    def __call__(self, table, rules, **kwargs):
        """
        table has been trimmed of extraneous columns.
        Topdown will use all columns to construct rules
        """
        self.rules = rules
        self.final_clusters = DiscreteBottomUp.__call__(self, table)
        return self.final_clusters

    def discrete_vals_iter(self, table):
        domain = table.domain
        for r in self.rules:
            partition = r.filter_table(table)
            if not len(partition):
                continue
            
            rmcols = [attr.name for attr in domain
                      if (attr.var_type == Orange.feature.Type.Discrete
                          and attr.name in self.cols)]
            cols = [col for col in self.cols if col not in rmcols]
            continuous_table = rm_attr_from_domain(partition, rmcols)

            partition_keys = {}
            for c in r.filter.conditions:
                if domain[c.position].var_type == Orange.feature.Type.Discrete:
                    partition_keys[domain[c.position].name] = map(int, c.values)

            yield partition_keys, cols, continuous_table


class Hybrid(object):
    def __init__(self, **kwargs):
        self.aggerr = kwargs['aggerr']
        self.topdown = HybridTopDown(**kwargs)

        kwargs['err_func'] = kwargs.get('err_func', self.aggerr.error_func)
        self.bottomup = HybridBottomUp(**kwargs)

        self.params = dict(kwargs)

        self.sample_cost = 0.
        self.kd_cost = 0.
        self.initclusters_cost = 0.
        self.merge_cost = 0.

        


    def __call__(self, table):
        start = time.time()
        rules = self.topdown(table)
        print 'rules'
        print '\n'.join(map(str, rules))

        self.bottomup.err_func.setup(table)

        all_rules = list(rules)
        if len(self.topdown.rules) > len(rules):
            all_rules.append(min(self.topdown.rules, key=lambda r: r.quality))

        self.final_clusters = self.bottomup(table, all_rules)
        self.all_clusters = self.bottomup.all_clusters

        self.cost = time.time() - start

        return self.final_clusters


