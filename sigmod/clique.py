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
from bottomup.bounding_box import *
from bottomup.cluster import *
from util import *
from basic import Basic
from merger import Merger
from grouper import Grouper, Blah

_logger = get_logger()
 


class MR(Basic):


    def __init__(self, *args, **kwargs):
        Basic.__init__(self, *args, **kwargs)
        self.best = []
        self.max_wait = kwargs.get('max_wait', 2 * 60 * 60) # 2 hours
        self.start = None
        self.stop = False
        self.n_rules_checked = 0
        self.naive = kwargs.get('naive', False)

        self.checkpoints = []

    def setup_tables(self, full_table, bad_tables, good_tables, **kwargs):
        Basic.setup_tables(self, full_table, bad_tables, good_tables, **kwargs)
        self.grouper = Grouper(full_table, self) 

        self.SCORE_ID = add_meta_column(
                chain([full_table], bad_tables, good_tables),
                'SCOREVAR' 
        )




    def set_params(self, **kwargs):
        self.cols = kwargs.get('cols', self.cols)
        self.params.update(kwargs)
        self.max_bad_inf = -1e1000000
        self.good_thresh = 0.0001
        self.granularity = kwargs.get('granularity', self.granularity)

    def make_rules(self, cur_groups):
        if cur_groups == None:
            new_groups = self.grouper.initial_groups()
        else:
            new_groups = self.grouper.merge_groups(cur_groups)

        rules = {}

        for attrs, groups in new_groups:
            start = time.time()
            for ro in self.grouper(attrs, groups):

                if self.max_wait:
                    self.n_rules_checked -= len(ro.rule.filter.conditions)
                    if self.n_rules_checked <= 0:
                        diff = time.time() - self.start
                        if not self.checkpoints or diff - self.checkpoints[-1][0] > 10:
                            if self.best:
                                best_rule = max(self.best, key=lambda r: r.inf).rule
                                self.checkpoints.append((diff, best_rule))
                        self.stop = diff > self.max_wait
                        self.n_rules_checked = 1000
                    if self.stop:
                        return


                yield attrs, ro
#            print "group by\t%s\t%.4f" % (str([attr.name for attr in attrs]), time.time()-start)



    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        rules = None
        self.best = []
        self.start = time.time()

        niters = 0 
        while not self.stop and (rules is None or rules):
            niters += 1
            print "========iter %d===========" % (niters)
            _logger.debug("=========iter %d=========", niters)
            besthash = hash(tuple(self.best))

            nadded = 0
            nnewgroups = 0
            new_rules = defaultdict(list)
            for attr, ro in self.make_rules(rules):
                if self.stop:
                    break
                nadded += self.top_k((ro,))
                if self.naive:
                    new_rules[attr] = [None]
                    nnewgroups += 1
                elif self.prune_rule(ro):
                    new_rules[attr].append(ro.group)
                    nnewgroups += 1
                ro.rule.__examples__ = None
                if nnewgroups % 10000 == 0:
                    print "# new groups\t", nnewgroups, '\t', time.time()-self.start, self.max_wait

            if not nadded: 
                break

            rules = new_rules

        self.cost_clique = time.time() - self.start



        self.best.sort(reverse=True)
        rules = [ro.rule for ro in self.best]
        return self.merge_rules(rules)

    def merge_rules(self, rules):
        fill_in_rules(rules, self.full_table, cols=self.cols)
        clusters = [Cluster.from_rule(r, self.cols) for r in rules]

        start = time.time()

        for c in clusters:
            c.error = self.influence(c) 
        clusters = filter(lambda c: c.error != -1e1000000, clusters)
        thresh = compute_clusters_threshold(clusters)
        is_mergable = lambda c: c.error >= thresh
        params = dict(self.params)
        params.update({'cols' : self.cols,
                       'influence' : lambda cluster: self.influence(cluster),
                       'is_mergable' : is_mergable})
        self.merger = Merger(**params)
        self.final_clusters = self.merger(clusters)
        self.all_clusters = clusters
        self.cost_merge = time.time() - start
         

        self.costs = {
                'cost_clique' : self.cost_clique,
                'cost_merge' : self.cost_merge
        }

        return self.final_clusters


    def influence(self, cluster):
        rule = cluster.to_rule(self.full_table, self.cols, cont_dists=self.cont_dists, disc_dists=self.disc_dists)
        return Basic.influence(self, rule)



    def prune_rules(self, rules):
        ret = defaultdict(set)
        for key, ros in rules.iteritems():
            for ro in ros:
                if self.prune_rule(ro):
                    ret[key].add(ro)
        return ret
    
    def prune_rule(self, ro):
        # update bad influence bounds
        self.max_bad_inf = max(self.max_bad_inf, ro.bad_inf)
        self.bad_thresh = max(self.bad_thresh, 0.01 * self.max_bad_inf)

        if ro.npts < self.min_pts:
            _logger.debug("%s\t%s", 'FALSE', str(ro))
            return False
        
        if (math.isnan(ro.bad_inf) or
            math.isnan(ro.good_inf) or
            math.isnan(ro.inf)):
            _logger.debug("%s\t%s", 'FALSE', str(ro))
            return False
        
        # check min bad influence
        if ro.bad_inf < self.bad_thresh:
            return False


        # assuming the best case (the good_stat was zero)
        # would the influence beat the min of best so far?
        if self.best and ro.best_inf <= max(self.best).inf:
            # if best tuple influence < rule influence:
            if ro.best_tuple_inf <= max(self.best).inf:
#                _logger.debug("%s\t%s", 'KKKFALSE', str(ro))
                return False
 
        # check max good influence
        if ro.good_inf < self.good_thresh:
            # TODO: can skip computing good_stats
            ro.good_skip = True

        #_logger.debug("%s\t%.4f\t%s", 'T', self.best and max(self.best).inf or 0, str(ro))
        return True


    def top_k(self, rules):

        n = 0
        best = self.best and max(self.best, key=lambda ro: ro.inf) or None
        for ro in rules:
            if len(self.best) >= self.max_bests:
                bound = best.inf - self.best[0].inf
                thresh = self.best[0].inf + bound * 0.02
                if ro.inf <= thresh:
                    continue
            if ro in self.best:
                continue
            if math.isnan(ro.inf):
                continue

            if not best or ro.inf > best.inf:
                n += 1            

            if len(self.best) < self.max_bests:
                heapq.heappush(self.best, ro)
            else:
                heapq.heapreplace(self.best, ro)
            
            best = best and max(best, ro) or ro
            _logger.debug(str(ro))

        return n



