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
from rangemerger import RangeMerger
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
        self.max_bests = 50
        self.max_complexity = kwargs.get('max_complexity', 3)

        self.checkpoints = []

        self.cost_clique = 0

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
              _logger.debug("wait %d > %d exceeded." % (diff, self.max_wait))
              return


          yield attrs, ro
#        print "group by\t%s\t%.4f" % (str([attr.name for attr in attrs]), time.time()-start)



    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        rules = None
        self.opts_per_iter = []
        self.best = []
        self.start = time.time()

        nseen = 0
        niters = 0 
        while niters < self.max_complexity and not self.stop and (rules is None or rules):
            niters += 1
            _logger.debug("=========iter %d=========", niters)
            besthash = hash(tuple(self.best))

            nadded = 0
            nnewgroups = 0
            new_rules = defaultdict(list)
            
            # for each combination of attributes
            # prune the groups that are less influential than the parent group's 
            #  

            for attr, ro in self.make_rules(rules):
                nseen += 1
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
                    pass
                    #print "# new groups\t", nnewgroups, '\t', time.time()-self.start, self.max_wait

            # all_rules = [rule for _, rule in self.make_rules(None)]
            # grouper = self.grouper
            # ef = self.bad_err_funcs[0]
            # bad = self.bad_tables[0]
            # attr = bad.domain[1]
            # groups = [g for g in grouper.initial_groups()]
            # groups = filter(lambda g: g[0][0] == attr, groups)
            # group = groups[0]
            # valid_groups = [g for g in group[1]]
            # valid_group = valid_groups[0]
            # table_inf = grouper.table_influence([attr], valid_groups, bad)
            # ef = self.bad_err_funcs[0]
            # foo = grouper._get_infs([table_inf], [ef], valid_group, True)
            # rows = table_inf.get(valid_group, [])


            if not nadded: 
                pass
#                break

            rules = new_rules
            if niters == 1:
                self.opts_per_iter.append(list(self.best))
            else:
                self.opts_per_iter.append(list(self.best[1:]))
                if prev_best and prev_best in self.opts_per_iter[-1]:
                    self.opts_per_iter[-1].remove(prev_best)
            self.best = [max(self.best)] if self.best else []
            prev_best = max(self.best) if self.best else None


        _logger.debug("finished, merging now")
        self.cost_clique = time.time() - self.start


        ret = []
        for bests in self.opts_per_iter:
            bests.sort(reverse=True)
            ret.extend(bests)# self.merge_rules(bests))
        clusters = map(self.blah_to_cluster, ret)
        clusters = self.merge_rules(clusters)
        clusters.sort(reverse=True)
        self.all_clusters = self.final_clusters = clusters
        return clusters

        self.best.sort(reverse=True)
        return self.merge_rules(self.best)

    def blah_to_cluster(self, blah):
        rule = blah.rule
        fill_in_rules([rule], self.full_table, self.cols)
        c = Cluster.from_rule(rule, self.cols)
        c.error = self.influence(c)
        return c


    def merge_rules(self, clusters):
        start = time.time()

        clusters = filter_bad_clusters(clusters)
        thresh = compute_clusters_threshold(clusters, nstds=0.)
        is_mergable = lambda c: c.error >= thresh
        is_mergable = lambda c: True
        params = dict(self.params)
        params.update({'cols' : self.cols,
                       'influence' : lambda cluster: self.influence(cluster),
                       'is_mergable' : is_mergable,
                       'c_range': self.c_range,
                       'use_mtuples' : False,
                       'learner' : self,
                       'partitions_complete' : False
                       })
        self.merger = RangeMerger(**params)
        #self.merger = Merger(**params)
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
        Basic.influence(self, rule)
        cluster.error = rule.quality
        cluster.inf_state = rule.inf_state
        return cluster.error



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
#        if ro.bad_inf < self.bad_thresh:
#            return False


        # assuming the best case (the good_stat was zero)
        # would the influence beat the best so far?
        if self.best and ro.best_inf <= max(self.best).inf:
            # if best tuple influence < rule influence:
            if ro.best_tuple_inf <= max(self.best).inf:
#                _logger.debug("%s\t%s", 'FALSE', str(ro))
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
                _logger.debug(str(ro))

            if len(self.best) < self.max_bests:
                heapq.heappush(self.best, ro)
            else:
                heapq.heapreplace(self.best, ro)
            
            best = best and max(best, ro) or ro

        return n



