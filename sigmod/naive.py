import time
import pdb
import sys
import Orange
import orange
import heapq
sys.path.extend(['.', '..'])

from itertools import chain
from collections import deque


from learners.cn2sd.rule import fill_in_rules
from learners.cn2sd.refiner import *
from score import QuadScoreSample7
from bottomup.bounding_box import *
from bottomup.cluster import *
from util import *

from util.misc import powerset
from basic import Basic

inf = 1e10000000
_logger = get_logger()

class Naive(Basic):
    def __init__(self, *args, **kwargs):
        Basic.__init__(self, *args, **kwargs)
        self.start = time.time()
        self.max_wait = kwargs.get('max_wait', 60*60*2) # 2 hours default
        self.n_rules_checked = 0
        self.stop = False

        self.checkpoints = []


    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)
        
        self.bests = []
        self.max_complexity = kwargs.get('max_complexity', self.max_complexity)
        self.granularity = kwargs.get('granularity', self.granularity)
        discretes = [attr for attr in full_table.domain 
                     if attr.name in self.cols and attr.var_type == Orange.feature.Type.Discrete]


        self.all_clauses = map(self.all_unit_clauses, self.cols)
        self.col_to_clauses = dict(zip(self.cols, self.all_clauses))
        base_rule = SDRule(self.full_table, None)            


        start = time.time()
        if discretes:
            max_card = max([len(attr.values) for attr in discretes])
            for m in xrange(1, max_card+1):
                if self.stop:
                    break
                self.foo(base_rule, max_card=m)
        else:
            self.foo(base_rule)
        self.cost = time.time() - start



        # given a rule = c1,..,cn
        # options for each clause c_i
        # 1) extend c_i to the left, to the right
        # 2) append a new clause
        rules = self.bests
        rules.sort(key=lambda r: r.quality, reverse=True)
        _logger.debug("best\n%s", "\n".join(map(lambda r: '%.4f\t%s' % (r.quality, str(r)), rules)))

        fill_in_rules(rules, full_table, cols=self.cols)
        clusters = [Cluster.from_rule(rule, self.cols, rule.quality) for rule in rules]
        self.all_clusters = clusters

        self.costs = {'cost' : self.cost}
        return clusters

    def max_card_in_conds(self, conds):
        lens = []
        for c in conds:
            attr = self.full_table.domain[c.position]
            if attr.var_type == Orange.feature.Type.Discrete:
                lens.append(len(attr.values))
        return lens and max(lens) or 0

    def foo(self, rule, max_card=None):
        for cols in powerset(self.cols):
            if not cols:
                continue
            if len(cols) > self.max_complexity:
                continue

            _logger.debug(str( cols))
            all_clauses = [self.get_all_clauses(col, max_card) for col in cols]
            for clauses in self.dfs(*all_clauses):
                new_rule = SDRule(rule.data, None, clauses, None)
                
                self.n_rules_checked -= len(clauses)
                if self.n_rules_checked <= 0:
                    diff = time.time() - self.start
                    if not self.checkpoints or diff - self.checkpoints[-1][0] > 10:
                        if self.bests:
                            best_rule = max(self.bests, key=lambda r: r.quality)
                            self.checkpoints.append((diff, best_rule))
                    self.stop = diff > self.max_wait
                    self.n_rules_checked = 100

                    _logger.debug( "%.4f\t%d", time.time() - self.start, self.n_rules_checked)
                    _logger.debug(str( new_rule))

                if self.stop:
                    return

                if max_card is not None and self.max_card_in_conds(clauses) < max_card:
                    continue

                new_rule.quality = self.influence(new_rule)
                new_rule.__examples__ = None
                if math.isnan(new_rule.quality) or new_rule.quality == -1e100000000:
                    continue


                if len(self.bests) < self.max_bests:
                    heapq.heappush(self.bests, new_rule)
                else:
                    heapq.heapreplace(self.bests, new_rule)

    def dfs(self, *iterables, **kwargs):
        q = kwargs.get('q', deque())
        if not iterables:
            yield q
            return

        for o in iterables[0]:
            q.append(o)
            for clauses in self.dfs(*iterables[1:], q=q):
                yield clauses
            q.pop()


