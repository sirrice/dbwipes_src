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



class RuleObj(object):
    def __init__(self, rule, mr):
        bad_err_funcs = mr.bad_err_funcs
        good_err_funcs = mr.good_err_funcs
        bad_tables = mr.bad_tables
        good_tables = mr.good_tables
        compute_stat = mr.compute_stat
        c = mr.c
        lamb = mr.lamb

        self.rule = rule
        self.bad_deltas, self.bad_counts = compute_stat(rule, bad_err_funcs, bad_tables)
        self.good_deltas, self.good_counts = compute_stat(rule, good_err_funcs, good_tables)
        self.good_stats = map(abs, self.good_deltas)
        self.good_skip = False


        self.bad_infs = [bd / ((bc+1)**c) for bd,bc in zip(self.bad_deltas, self.bad_counts) if bc]
        self.good_infs = [gd / ((gc+1)**c) for gd,gc in zip(self.good_deltas, self.good_counts) if gc]
        self.bad_inf = self.bad_infs and np.mean(self.bad_infs) or 0
        self.good_inf = self.good_infs and max(self.good_infs) or 0
        self.inf = lamb * self.bad_inf - (1.-lamb) * self.good_inf
        self.best_inf = self.bad_inf

        self.npts = sum(self.bad_counts + self.good_counts)
        self.good_npts = sum(self.good_counts)
        self.bad_npts = sum(self.bad_counts)
        self.mean_pts = np.mean(self.bad_counts + self.good_counts)

        self.rule.quality = self.inf
        if not math.isnan(self.inf):
            print  self
    
    def __str__(self):
        return 'inf %.4f\tpts %d/%d\tr/g %.4f - %.4f\t%s' % (self.inf, sum(self.bad_counts), sum(self.good_counts), self.bad_inf, self.good_inf, self.rule) 

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

    def __eq__(self, o):
        return hash(self) == hash(o)


    def __cmp__(self, o):
        if self.inf > o.inf:
            return 1
        if self.inf == o.inf:
            return 0
        return -1



class MR(Basic):

    def set_params(self, **kwargs):
        self.cols = kwargs.get('cols', self.cols)
        self.params.update(kwargs)
        self.max_bad_inf = -1e1000000
        self.good_thresh = 0.0001


    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

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
            print '\n'.join(map(lambda ro: '\tr/g %.4f - %.4f\tinf %.4f\tpts %d/%d\t%s' % (ro.bad_inf, ro.good_inf, ro.inf, ro.bad_npts, ro.good_npts, str(ro.rule)), self.best))

            
            if not nadded:
                break


            rules = pruned_rules



        self.best.sort(reverse=True)
        rules = [ro.rule for ro in self.best]
        fill_in_rules(rules, self.full_table, cols=self.cols)
        clusters = [Cluster.from_rule(r, self.cols) for r in rules]

        for c in clusters:
            c.error = self.influence(c) 
        thresh = compute_clusters_threshold(clusters)
        is_mergable = lambda c: c.error >= thresh
        params = dict(self.params)
        params.update({'cols' : self.cols,
                       'influence' : lambda cluster: self.influence(cluster),
                       'is_mergable' : is_mergable})
        self.merger = Merger(**params)
        self.final_clusters = self.merger(clusters)
        self.all_clusters = clusters


        print "========Post Merge=========="
        ros = [RuleObj(r, self) for r in clusters_to_rules(self.final_clusters, self.cols, self.full_table)]
        ros.sort(key=lambda ro: ro.inf, reverse=True)
        print '\n'.join(map(str, ros))


        return self.final_clusters


    def influence(self, cluster):
        rule = cluster.to_rule(self.full_table, self.cols, cont_dists=self.cont_dists, disc_dists=self.disc_dists)
        return Basic.influence(self, rule)



    def prune_rules(self, rules):
        ret = {}
        for key, ros in rules.iteritems():
            ros = filter(self.prune_rule, ros)
            ret[key] = set(ros)
        return ret
    
    def prune_rule(self, ro):
        # update bad influence bounds
        self.max_bad_inf = max(self.max_bad_inf, ro.bad_inf)
        self.bad_thresh = max(self.bad_thresh, 0.01 * self.max_bad_inf)

        if ro.npts < self.min_pts:
            return False
        
        if (math.isnan(ro.bad_inf) or
            math.isnan(ro.good_inf) or
            math.isnan(ro.inf)):
            return False
        
        # check min bad influence
        if ro.bad_inf < self.bad_thresh:
            return False

        # assuming the best case (the good_stat was zero)
        # would the influence beat the min of best so far?
        if self.best and ro.best_inf < max(self.best).inf:
            return False
 
        # check max good influence
        if ro.good_inf < self.good_thresh:
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


            ro = RuleObj(rule,self)
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
                    newro = RuleObj(newrule, self)
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


                ro = RuleObj(rule,self)


                if ro.npts in (ro1.npts, ro2.npts):
                    continue
                
                yield ro
