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
from merger.merger import Merger
from merger.reexec import ReexecMerger
from util import *



class RuleObj(object):
    def __init__(self, rule, bad_err_funcs, good_err_funcs, bad_tables, good_tables):
        self.rule = rule
        self.bad_stats, self.bad_counts = self.compute_stat(bad_err_funcs, bad_tables)
        self.good_stats, self.good_counts = self.compute_stat(good_err_funcs, good_tables)
        self.good_stats = map(abs, self.good_stats)
        self.good_skip = False

        # compute diff
        f = lambda counts: (np.mean(counts))
        self.bad_stat = np.mean(self.bad_stats)
        self.good_stat = self.good_stats and max(self.good_stats) or 0
        self.npts = max(self.bad_counts + self.good_counts)
        self.mean_pts = np.mean(self.bad_counts + self.good_counts)
        self.bad_inf = self.bad_stat# / (1+f(self.bad_counts))
        self.good_inf = self.good_stat# / (1+f(self.good_counts))
        self.inf =  self.bad_inf - self.good_inf
        self.best_inf = self.bad_inf

        self.rule.quality = self.inf
        if not math.isnan(self.inf):
            print '%f\t%f\t%f\t%f\t%s' % (self.inf, self.npts, self.bad_stat, self.good_stat, self.rule) 

    def __hash__(self):
        return hash(self.rule)

    def compute_stat(self, err_funcs, tables):
        datas = map(self.rule.filter_table, tables)
        infs = []
        for ef, data in zip(err_funcs, datas):
            arr = data.to_numpyMA('ac')[0]
            inf = ef(arr.data)
            infs.append(inf)
        return infs, map(len, datas)


    def __cmp__(self, o):
        if self.inf > o.inf:
            return 1
        if self.inf == o.inf:
            return 0
        return -1



class CLIQUE(object):
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

        

        self.scorer_cost = 0.
        self.merge_cost = 0.

        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        self.cols = kwargs.get('cols', self.cols)
        self.params.update(kwargs)


    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.full_table = full_table
        self.bad_tables = bad_tables
        self.good_tables = good_tables
        self.bad_err_funcs = [self.err_func.clone() for t in bad_tables]
        self.good_err_funcs = [self.err_func.clone() for t in good_tables]

        for ef, t in zip(chain(self.bad_err_funcs, self.good_err_funcs),
                         chain(bad_tables, good_tables)):
            ef.setup(t)

        self.max_bad_stat = -1e100000000


        rules = None
        self.best = []
        
        while rules is None or rules:

            new_rules = self.make_rules(rules)
            nadded = self.top_k(new_rules)
            pruned_rules = self.prune_rules(new_rules)

            print "bad thresh\t%f" % self.bad_thresh
            print "n added\t%d" % nadded
            print "n rules\t%d" % (sum(map(len, pruned_rules.values())))
            self.best.sort()
            print '\n'.join(map(lambda ro: '\t%f\t%s' % (ro.inf, str(ro.rule)), self.best))

            
            if not nadded:
                break


            rules = pruned_rules

        self.best.sort(reverse=True)
        rules = [ro.rule for ro in self.best]
        self.err_func = self.bad_err_funcs[0]
        fill_in_rules(rules, self.full_table, cols=self.cols)
        self.all_clusters = [Cluster.from_rule(r, self.cols) for r in rules]
        self.all_clusters.sort(key=lambda c: c.error, reverse=True)
        return self.all_clusters

        thresh = compute_clusters_threshold(self.all_clusters)
        is_mergable = lambda c: c.error >= thresh
        print "threshold", thresh


        start = time.time()
        params = dict(self.params)
        params.update({'cols' : self.cols,
                       'full_table' : full_table,
                       'bad_tables' : self.bad_tables,
                       'good_tables' : self.good_tables,
                       'bad_err_funcs' : self.bad_err_funcs,
                       'good_err_funcs' : self.good_err_funcs,
                       'err_func' : self.err_func})
        self.merger = ReexecMerger(**params)
        self.final_clusters = self.merger(self.all_clusters, is_mergable=is_mergable)
        self.final_clusters.sort(key=lambda c: c.error, reverse=True)
        self.merge_cost = time.time() - start

        final_rules = clusters_to_rules(self.final_clusters, 
                self.cols, full_table)
        print "\n============Besties==========="
        for rule in final_rules:
            print "%f\t%s" % (rule.quality, str(rule))


        return self.final_clusters





        return self.all_clusters


    def prune_rules(self, rules):
        ret = {}
        for key, ros in rules.iteritems():
            ros = filter(self.prune_rule, ros)
            ret[key] = set(ros)
        return ret
    
    def prune_rule(self, ro):
        # update bad influence bounds
        self.max_bad_stat = max(self.max_bad_stat, ro.bad_stat)
        self.bad_thresh = max(self.bad_thresh, 0.01 * self.max_bad_stat)

        if ro.npts < self.min_pts:
            return False
        
        if (math.isnan(ro.bad_stat) or
            math.isnan(ro.good_stat) or
            math.isnan(ro.inf)):
            return False
        
        # check min bad influence
        if ro.bad_stat < self.bad_thresh:
            return False

        # assuming the best case (the good_stat was zero)
        # would the influence beat the min of best so far?
        if self.best and ro.best_inf < max(self.best).inf:
            return False
 
        # check max good influence
        if ro.good_stat < self.good_thresh:
            # TODO: can skip computing good_stats
            ro.good_skip = True

        return True


    def top_k(self, rules):

        n = 0
        for ro in chain(*rules.values()):
            if len(self.best) >= self.max_bests:
                bound = self.best[1].inf - self.best[0].inf
                thresh = self.best[0].inf + bound * 0.05
                if ro.inf <= thresh:
                    continue
            if ro in self.best:
                continue
            if math.isnan(ro.inf):
                continue

            n += 1            
            if len(self.best) < self.max_bests:
                heapq.heappush(self.best, ro)
            else:
                heapq.heapreplace(self.best, ro)

        return n


            
    def make_rules(self, old_rules):
        """
        Merge rules in old_rules to compute next round of rules
        or create 1D partitions of each attribute
        """
        rules = defaultdict(set)

        def recurse_disc_rule(attr, rule):
            """
            Recursively partition multivalued discrete attributes if
            its worth it
            """


            ro = RuleObj(rule,
                         self.bad_err_funcs,
                         self.good_err_funcs,
                         self.bad_tables,
                         self.good_tables)

            if not self.prune_rule(ro):
                return set([ro])
            
            c = rule.filter.conditions[0]
            var_type = rule.data.domain[c.position].var_type

            if (var_type == Orange.feature.Type.Discrete):
                if len(c.values) == 1:
                    return [ro]
            
                refiner = BeamRefiner(attrs=[attr], fanout=10)
                ret = set()
                for _, newrule in refiner(rule):
                    ret.update(recurse_disc_rule(attr, newrule))
                return ret
            else:
                if len(rule.data) < self.min_pts:
                    return [ro]
                return [ro]

                # XXX: figure out this logic!

                refiner = BeamRefiner(attrs=[attr], fanout=2)
                ret = set()
                for _, newrule in refiner(rule):
                    newro = RuleObj(newrule,
                                 self.bad_err_funcs,
                                 self.good_err_funcs,
                                 self.bad_tables,
                                 self.good_tables)
                    ret.update(recurse_disc_rule(attr, newrule))

        
        if old_rules is None:
            base_rule = SDRule(self.full_table, None)            
            refiner = BeamRefiner(attrs=self.cols, fanout=10)
            #refiner = BeamRefiner(attrs=['recipient_nm'], fanout=30)        

            
            for attr, rule in refiner(base_rule):
                ros = recurse_disc_rule(attr, rule)
                #self.top_k({None:ros})
                ros = filter(self.prune_rule, ros)
                rules[(attr,)].update(ros)

        else:
            attrs = old_rules.keys()
            for a_idx, attr1 in enumerate(attrs):
                for attr2 in attrs[a_idx+1:]:
                    merged_attrs = set(attr1).union(attr2)
                    max_attrs_len = max(len(attr1), len(attr2))
                    if len(merged_attrs) == max_attrs_len:
                        continue
                        
                    
                    a1rules, a2rules = old_rules[attr1], old_rules[attr2]

                    for ro in self.merge_dims(a1rules, a2rules):
                        key = ro.rule.attributes

                        #self.top_k({None:(ro,)})
                        if self.prune_rule(ro):
                            rules[key].add(ro)
                        
        return rules

    def can_merge(self, r1, r2):
        attrs1 = r1.attributes
        attrs2 = r2.attributes
        iattrs = set(r1.attributes).intersection(r2.attributes)
        if (len(iattrs) != len(attrs1) - 1 or
            len(iattrs) != len(attrs2) - 1):
            return False

        def rule_dict(r):
            domain = r1.data.domain
            return dict([(domain[c.position].name, r.condToString(c))
                         for c in r.filter.conditions])

        d1, d2 = rule_dict(r1), rule_dict(r2)

        for attr in iattrs:
            if d1[attr] != d2[attr]:
                return False
        return True

            
    def merge_dims(self, a1rules, a2rules):
        seen = set()
        for ro1 in a1rules:
            for ro2 in a2rules:
                if ro1.rule.attributes == ro2.rule.attributes:
                    continue
                
                rule1, rule2 = ro1.rule, ro2.rule

                if not self.can_merge(rule1, rule2):
                    continue

                # transform scales and value indexs from rule2 -> rule1 conditions
                # skip conditions that share the same attribute with rule1
                attrs1 = rule1.attributes
                attrs2 = rule2.attributes
                iattrs = set(rule2.attributes).difference(rule1.attributes)
                domain1, domain2 = rule1.data.domain, rule2.data.domain
                filter_func = lambda c: domain2[c.position].name in iattrs
                conds = []
                for c in filter(filter_func, rule2.filter.conditions):
                    attr = domain2[c.position]
                    var_type = attr.var_type
                    if var_type == Orange.feature.Type.Discrete:
                        vals = [orange.Value(domain1[attr],
                                             domain2[attr].values[int(vidx)])
                                             for vidx in c.values]
                        c = orange.ValueFilter_discrete(
                            position = domain1.index(domain1[attr]),
                            values = vals
                            )
                    conds.append(c)
                
                conds = list(rule1.filter.conditions) + conds
                rule = SDRule(rule1.data, rule1.targetClass, conds, rule1.g)
                if hash(str(rule)) in seen:
                    continue
                seen.add(hash(str(rule)))


                ro = RuleObj(rule,
                             self.bad_err_funcs,
                             self.good_err_funcs,
                             self.bad_tables,
                             self.good_tables)


                if ro.npts == max(ro1.npts, ro2.npts):
                    continue
                
                yield ro
