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

_logger = get_logger()


class Blah(object):
    def __init__(self, attrs, group, bad_deltas, bad_counts, good_deltas, good_counts, maxinf, mr, grouper):
        self.attrs = attrs
        self.grouper = grouper
        self.group = group
        self.bad_deltas = bad_deltas
        self.bad_counts = bad_counts
        self.good_deltas = good_deltas
        self.good_counts = good_counts
        self.mr = mr
        c = mr.c
        l = mr.l

        self.bad_infs = [bd / ((bc+1)**c) for bd,bc in zip(self.bad_deltas, self.bad_counts) if bc]
        self.good_infs = [gd / ((gc+1)**c) for gd,gc in zip(self.good_deltas, self.good_counts) if gc]
        self.bad_inf = l * (self.bad_infs and np.mean(self.bad_infs) or 0)
        self.good_inf = (1. - l) * (self.good_infs and max(self.good_infs) or 0)
        self.inf = self.bad_inf - self.good_inf
        self.best_inf = self.bad_inf
        self.best_tuple_inf = maxinf

        self.npts = sum(self.bad_counts + self.good_counts)
        self.good_npts = sum(self.good_counts)
        self.bad_npts = sum(self.bad_counts)
        self.mean_pts = np.mean(self.bad_counts + self.good_counts)

        self._rule = None



    def __rule__(self):
        if self._rule:
            return self._rule
        conds = []
        for attr, gid in zip(self.attrs, self.group):
            if attr.var_type ==  Orange.feature.Type.Discrete:
                vals = [orange.Value(attr, v) for v in self.grouper.id2vals[attr][gid]]
                conds.append(
                        orange.ValueFilter_discrete(
                            position = self.grouper.data.domain.index(attr),
                            values = vals
                        )
                )
            else:
                vals = self.grouper.id2vals[attr][gid]
                minv, maxv = vals[0], vals[1]
                conds.append(
                        Orange.data.filter.ValueFilterContinuous(
                    oper=orange.ValueFilter.Between,
                    position = self.grouper.data.domain.index(attr),
                    min=minv,
                    max=maxv)
                        )
        self._rule = SDRule(self.grouper.data, None, conds, None)
        self._rule.quality = self.inf
        return self._rule
    rule = property(__rule__)


    def __str__(self):
        return 'inf %.4f\tpts %d/%d\tr/g %.4f - %.4f\t%s' % (self.inf, sum(self.bad_counts), sum(self.good_counts), self.bad_inf, self.good_inf, self.rule) 



    def __hash__(self):
        return hash(str(self.group))

    def __eq__(self, o):
        return hash(self) == hash(o)


    def __cmp__(self, o):
        if self.inf > o.inf:
            return 1
        if self.inf == o.inf:
            return 0
        return -1



class RuleObj(object):
    def __init__(self, rule, mr):
        bad_err_funcs = mr.bad_err_funcs
        good_err_funcs = mr.good_err_funcs
        bad_tables = mr.bad_tables
        good_tables = mr.good_tables
        compute_stat = mr.compute_stat
        c = mr.c
        l = mr.l

        self.rule = rule
        self.bad_deltas, self.bad_counts = compute_stat(rule, bad_err_funcs, bad_tables)
        self.good_deltas, self.good_counts = compute_stat(rule, good_err_funcs, good_tables)
        self.good_stats = map(abs, self.good_deltas)
        self.good_skip = False


        self.bad_infs = [bd / ((bc+1)**c) for bd,bc in zip(self.bad_deltas, self.bad_counts) if bc]
        self.good_infs = [gd / ((gc+1)**c) for gd,gc in zip(self.good_deltas, self.good_counts) if gc]
        self.bad_inf = l * (self.bad_infs and np.mean(self.bad_infs) or 0)
        self.good_inf = (1. - l) * (self.good_infs and max(self.good_infs) or 0)
        self.inf = self.bad_inf - self.good_inf
        self.best_inf = self.bad_inf

        self.npts = sum(self.bad_counts + self.good_counts)
        self.good_npts = sum(self.good_counts)
        self.bad_npts = sum(self.bad_counts)
        self.mean_pts = np.mean(self.bad_counts + self.good_counts)

        self.rule.quality = self.inf
    
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


class Grouper(object):
    def __init__(self, table, mr):
        self.data = table
        self.mr = mr
        self.ddists = None
        self.bdists = None
        self.gbids = {}
        self.id2vals = {}
        self.gbfuncs = {}
        self.mappers = {}
        self.setup_functions()

    def setup_functions(self):
        domain = self.data.domain
        ddists = Orange.statistics.distribution.Domain(self.data)
        self.ddists = ddists
        bdists = Orange.statistics.basic.Domain(self.data)
        self.bdists = bdists
        gbfuncs = {}
        gbids = {}
        id2vals = {}

        for idx, attr in enumerate(self.data.domain):
            if attr.name not in self.mr.cols:
                continue
            if attr.var_type == Orange.feature.Type.Discrete:
                groups = self.create_discrete_groups(attr, idx, ddists[idx].keys())
                mapper = {}
                for idx, group in enumerate(groups):
                    for val in group:
                        mapper[val] = idx
                self.mappers[attr] = mapper

                f = lambda v: self.mappers[v.variable].get(v.value, len(groups))
                n = len(groups)
                ranges = groups
            else:
                dist = bdists[idx]
                maxv, minv = dist.max, dist.min
                if maxv == minv:
                    f = lambda v: 0.
                    n = 1
                    ranges = [(minv, maxv)]
                else:
                    block = (maxv - minv) / float(self.mr.granularity)
                    f = lambda v: math.floor((v-minv) / block)
                    ranges = [(minv + i*block, minv + (i+1)*block) for i in xrange(self.mr.granularity)]
                    n = self.mr.granularity

            gbfuncs[attr] = f
            gbids[attr] = n
            id2vals[attr] = dict(enumerate(ranges))

        self.gbfuncs = gbfuncs
        self.gbids = gbids
        self.id2vals = id2vals
        print gbids

    def create_discrete_groups(self, attr, pos, vals):
        return [(val,) for val in vals]
        if len(vals) == 1:
            return (vals,)

        rule = SDRule(self.data, None, [orange.ValueFilter_discrete(
            position = pos, 
            values = [orange.Value(attr,v) for v in vals]
        )], None)
        ro = RuleObj(rule, self.mr)

        if not self.mr.prune_rule(ro):
            return (vals,)

        ret = []
        for newvals in block_iter(vals, 2):
            ret.extend(self.create_discrete_groups(attr, pos, newvals))
        return ret



    def __call__(self, attrs, valid_groups):
        valid_groups = set(valid_groups)
        start = time.time()
        bad_table_rows = []
        good_table_rows = []
        for table in self.mr.bad_tables:
            bad_table_rows.append(self.table_influence(attrs, valid_groups, table))
        for table in self.mr.good_tables:
            good_table_rows.append(self.table_influence(attrs, valid_groups, table))
        print "scan time\t", (time.time() - start)

        def get_infs(all_table_rows, err_funcs, g, bmaxinf):
            ret = []
            counts = []
            maxinf = None
            for idx, ef, table_rows in zip(range(len(err_funcs)), err_funcs, all_table_rows):
                rows = table_rows.get(g, [])
                if rows:
                    if bmaxinf:
                        cur_maxinf = self.influence_tuple(max(rows, key=lambda row: self.influence_tuple(row, ef)), ef)
                        if not maxinf or cur_maxinf > maxinf:
                            maxinf = cur_maxinf

                    ret.append(ef(rows))
                    counts.append(len(rows))
            return ret, counts, maxinf


        ret = []
        start = time.time()
        for g in valid_groups:
            bds, bcs, maxinf = get_infs(bad_table_rows, self.mr.bad_err_funcs, g, True)
            gds, gcs, _ = get_infs(good_table_rows, self.mr.good_err_funcs, g, False)
            if not bcs:
                continue
            ret.append(Blah(attrs, g, bds, bcs, gds, gcs, maxinf, self.mr, self))
        print "comp infs\t", (time.time() - start)
        return ret


    def influence_tuple(self, row, ef):
        if row[self.mr.SCORE_ID].value == -1e10000000000:
            influence = ef((row,))
            row[self.mr.SCORE_ID] = influence
        return row[self.mr.SCORE_ID].value



    def table_influence(self, attrs, valid_groups, table):
        groups = defaultdict(list)
        for row in table:
            group = tuple([self.gbfuncs[attr](row[attr]) for attr in attrs])
            if group in valid_groups:
                groups[group].append(row)
        return groups

    def initial_groups(self):
        ret = defaultdict(set)
        for attr, n in self.gbids.items():
            for i in xrange(n):
                ret[(attr,)].add((i,))
        return ret




    def merge_groups(self, prev_groups):
        """
        prev_groups: attributes -> groups
        attributes are sorted
        group: attr -> idx
        """
        start = time.time()
        attrs_list = prev_groups.keys()
        ret = {}
        for a_idx, attrs1 in enumerate(attrs_list):
            for attrs2 in attrs_list[a_idx+1:]:
                merged_attrs = tuple(sorted(set(attrs1).union(attrs2)))
                if len(merged_attrs) != len(attrs1)+1:
                    continue
                intersecting = tuple(sorted(set(attrs1).intersection(attrs2)))
                ret[merged_attrs] = list()

                for group in self.fulljoin(intersecting, merged_attrs, 
                                           prev_groups[attrs1], prev_groups[attrs2]):
                    ret[merged_attrs].append( group)
        #print "merge groups: ", (time.time() - start)
        return ret

    def fulljoin(self, inter, union, groups1, groups2):
        interf = lambda g: tuple([g[k] for k in inter])
        unionf = lambda g: tuple([g[k] for k in union])
        seen = set()
        ret = []
        for g1 in groups1:
            for g2 in groups2:

                if interf(g1) != interf(g2):
                    continue
                
                newg = list()
                for attr in union:
                    if attr in g1:
                        newg.append(g1[attr])
                    else:
                        newg.append(g2[attr])
                newg = tuple(newg)
                if newg in seen:
                    continue
                seen.add(newg)
        return seen

             


 


class MR(Basic):


    def __init__(self, *args, **kwargs):
        Basic.__init__(self, *args, **kwargs)
        self.best = []

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

    def make_rules(self, cur_rules):
        if cur_rules == None:
            new_groups = self.grouper.initial_groups()
        else:
            cur_groups = {}
            for attrs, rules in cur_rules.items():
                cur_groups[attrs] = [dict(zip(ro.attrs, ro.group)) for ro in rules]
            new_groups = self.grouper.merge_groups(cur_groups)

        rules = {}

        for attrs, groups in new_groups.items():
            start = time.time()
            rules[attrs] = self.grouper(attrs, groups)
            #print "group by\t%s\t%.4f" % (str([attr.name for attr in attrs]), time.time()-start)
        return rules



    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        rules = None
        self.best = []
        start = time.time()
        
        while rules is None or rules:
            besthash = hash(tuple(self.best))
            
            new_rules = self.make_rules(rules)
            nadded = self.top_k(chain(*new_rules.values()))
            pruned_rules = self.prune_rules(new_rules)


            _logger.debug("bad thresh\t%f" , self.bad_thresh)
            _logger.debug("n rules\t%d" , (sum(map(len, pruned_rules.values()))))
            self.best.sort()
            f = lambda ro: '\tr/g %.4f - %.4f\tinf %.4f\tpts %d/%d\t%s' % (ro.bad_inf, ro.good_inf, ro.inf, ro.bad_npts, ro.good_npts, str(ro.rule))
            _logger.debug('\n'.join(map(f, self.best)))

            
            if not nadded: 
                break


            rules = pruned_rules

        self.cost_clique = time.time() - start



        self.best.sort(reverse=True)
        rules = [ro.rule for ro in self.best]
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


        _logger.debug( "========Post Merge==========")
        ros = [RuleObj(r, self) for r in clusters_to_rules(self.final_clusters, self.cols, self.full_table)]
        ros.sort(key=lambda ro: ro.inf, reverse=True)
        _logger.debug( '\n'.join(map(str, ros))) 

        self.costs = {'cost_clique' : self.cost_clique,
                'cost_merge' : self.cost_merge}

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
        if self.best and ro.best_inf < max(self.best).inf:
            # if best tuple influence < rule influence:
            if ro.best_tuple_inf < max(self.best).inf:
                _logger.debug("%s\t%s", 'KKKFALSE', str(ro))
                return False
 
        # check max good influence
        if ro.good_inf < self.good_thresh:
            # TODO: can skip computing good_stats
            ro.good_skip = True

        #_logger.debug("%s\t%.4f\t%s", 'T', self.best and max(self.best).inf or 0, str(ro))
        return True


    def top_k(self, rules):

        n = 0
        for ro in rules:
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



