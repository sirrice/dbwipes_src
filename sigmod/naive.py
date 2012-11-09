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

from util.misc import powerset
from basic import Basic

inf = 1e10000000

class Naive(Basic):


    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)
        
        self.bests = []
        self.max_complexity = kwargs.get('max_complexity', self.max_complexity)
        self.granularity = kwargs.get('granularity', self.granularity)


        self.all_clauses = map(self.all_unit_clauses, self.cols)
        self.col_to_clauses = dict(zip(self.cols, self.all_clauses))

        # given a rule = c1,..,cn
        # options for each clause c_i
        # 1) extend c_i to the left, to the right
        # 2) append a new clause

        base_rule = SDRule(self.full_table, None)            
        start = time.time()
        self.dfs(base_rule)
        self.cost = time.time() - start

        rules = self.bests

        fill_in_rules(rules, full_table, cols=self.cols)
        clusters = [Cluster.from_rule(rule, self.cols, rule.quality) for rule in rules]
        self.all_clusters = clusters

        self.costs = {'cost' : self.cost}
        return clusters


    def dfs(self, rule, colidx=0):
        if rule.complexity >= self.max_complexity:
            return

        if colidx >= len(self.cols):
            return

        for clause in self.get_all_clauses(self.cols[colidx]):
            conds = rule.filter.conditions[:]
            conds.append(clause)
            new_rule = SDRule(rule.data, rule.targetClass, conds, rule.g)

            new_rule.quality = self.influence(new_rule)
            if math.isnan(new_rule.quality) or new_rule.quality == -1e100000000:
                continue


            if len(self.bests) < self.max_bests:
                heapq.heappush(self.bests, new_rule)
            else:
                heapq.heapreplace(self.bests, new_rule)

            self.dfs(new_rule, colidx=colidx+1)

