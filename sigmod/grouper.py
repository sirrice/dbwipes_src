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
from crange import r_lte
from basic import Basic
from merger import Merger
from errfunc import compute_bad_inf

_logger = get_logger()


class Blah(object):
  def __init__(self, attrs, group, bad_deltas, bad_counts, 
      good_deltas, good_counts, maxinf, mr, grouper):

    self.attrs = attrs
    self.grouper = grouper
    self.group = group
    self.inf_state = (bad_deltas, bad_counts, good_deltas, good_counts)
    self.mr = mr
    self.c_range = list(mr.c_range)

    c = mr.c
    l = mr.l

    good_infs = [abs(gd) for gd,gc in zip(good_deltas, good_counts) if gc]
    self.good_inf = (1. - l) * (good_infs and max(good_infs) or 0)
    self.good_skip = False # subpredicates can skip computing good_inf

    self.bad_inf_func = self.create_bad_inf_func(l)
    self.inf_func = self.create_inf_func(l, self.good_inf)
    self.bad_inf = self.bad_inf_func(c)

    self.inf = self.bad_inf - self.good_inf
    self.best_inf = self.bad_inf
    self.best_tuple_inf = l * maxinf
    self.maxinf = maxinf
    self.npts = sum(bad_counts) + sum(good_counts)

    self._rule = None

  def create_bad_inf_func(self, l):
    bds, bcs, gds, gcs = self.inf_state
    bds = [bd for bd, bc in zip(bds, bcs) if bc]
    bcs = filter(bool, bcs)
    pairs = zip(bds, bcs)
    f = lambda c: l*compute_bad_score(bds, bcs, c)
    return f

  def create_inf_func(self, l, ginf):
    badf = self.create_bad_inf_func(l)
    g = lambda c: (1.-l)*ginf + l*badf(c)
    return g

  def dominated_by(self, o, c_range=None):
    if not c_range: c_range = self.c_range
    mybads = map(self.bad_inf_func, c_range)
    oinfs = map(o.inf_func, c_range)
    if r_lte(mybads, oinfs):
      if self.best_tuple_inf <= min(oinfs):
        return True
    return False


  def clone(self):
    args = [self.attrs, self.group] 
    args.extend(self.inf_state)
    args.extend((self.maxinf, self.mr, self.grouper))
    ret = Blah(*args)
    ret._rule = self._rule.clone()
    return ret


  def __rule__(self):
    if self._rule: return self._rule

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
            max=maxv
          )
        )
    self._rule = SDRule(self.grouper.data, None, conds, None)
    self._rule.quality = self.inf

    return self._rule
  rule = property(__rule__)


  def __str__(self):
    args = (
      self.inf, 
      sum(self.inf_state[1]), 
      sum(self.inf_state[3]), 
      self.bad_inf, self.good_inf, 
      self.rule
    ) 
    return 'inf %.4f\tpts %d/%d\tr/g %.4f - %.4f\t%s' % args

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

                f = lambda v: int(self.mappers[v.variable].get(v.value, len(groups)))
                n = len(groups)
                ranges = groups
            else:
                dist = bdists[idx]
                maxv, minv = dist.max, dist.min
                if maxv == minv:
                    f = lambda v: 0
                    n = 1
                    ranges = [(minv, maxv)]
                else:
                    def make_func(minv, block):
                        def f(v):
                            return int(math.floor((v-minv)/block))
                        return f
                    block = (maxv - minv) / float(self.mr.granularity)
                    f = make_func(minv, block)
                    ranges = [(minv + i*block, minv + (i+1)*block) for i in xrange(self.mr.granularity+1)]
                    n = self.mr.granularity+1
                    print attr, block, minv, maxv

            gbfuncs[attr] = f
            gbids[attr] = n
            id2vals[attr] = dict(enumerate(ranges))

        self.gbfuncs = gbfuncs
        self.gbids = gbids
        self.id2vals = id2vals

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


    def _get_infs(self, all_table_rows, err_funcs, g, bmaxinf):
      """
      Args:
        all_table_rows: list, each item is a dict of
                        group -> rows in that group (e.g., predicate)
        g: the group
        bmaxinf: want max inf of individual rows
      """
      ret = []
      counts = []
      maxinf = -1e1000000000000
      iterator = zip(enumerate(err_funcs), all_table_rows)
      for (idx, ef), table_rows in iterator:
        rows = table_rows.get(g, [])
        if not rows:
          continue
        
        if bmaxinf:
          for row in rows:
            curinf = self.influence_tuple(row, ef)
            maxinf = max(maxinf, curinf) 

        ret.append(ef(rows))
        counts.append(len(rows))
      return ret, counts, maxinf

    def influence_tuple(self, row, ef):
      if row[self.mr.SCORE_ID].value == -1e10000000000:
        influence = ef((row,))
        row[self.mr.SCORE_ID] = influence
      return row[self.mr.SCORE_ID].value

    def groups_by_attrs(self, attrs, valid_groups, table):
      """scan table once and group tuples by their respective groups"""
      groups = defaultdict(list)
      for row in table:
        group = tuple([self.gbfuncs[attr](row[attr]) for attr in attrs])
        if group in valid_groups:
          groups[group].append(row)
      return groups


    def __call__(self, attrs, valid_groups):
      valid_groups = set(valid_groups)
      bad_table_rows = []
      good_table_rows = []
      for table in self.mr.bad_tables:
        bad_table_rows.append(self.groups_by_attrs(attrs, valid_groups, table))
      for table in self.mr.good_tables:
        good_table_rows.append(self.groups_by_attrs(attrs, valid_groups, table))


      for g in valid_groups:
        bds, bcs, maxinf = self._get_infs(bad_table_rows, self.mr.bad_err_funcs, g, True)
        gds, gcs, _ = self._get_infs(good_table_rows, self.mr.good_err_funcs, g, False)
        if not bcs:
          continue
        yield Blah(attrs, g, bds, bcs, gds, gcs, maxinf, self.mr, self)




    def initial_groups(self):
        for attr, n in self.gbids.items():
            yield (attr,), ((i,) for i in xrange(n))




    def merge_groups(self, prev_groups):
        """
        prev_groups: attributes -> groups
        attributes are sorted
        group: attr -> idx
        """
        start = time.time()
        attrs_list = prev_groups.keys()
        for a_idx, attrs1 in enumerate(attrs_list):
            sattrs1 = set(attrs1)
            pgroup1 = [dict(zip(attrs1, g)) for g in prev_groups[attrs1]]
            for attrs2 in attrs_list[a_idx+1:]:
                pgroup2 = [dict(zip(attrs2, g)) for g in prev_groups[attrs2]]

                sattrs2 = set(attrs2)
                merged_attrs = tuple(sorted(sattrs1.union(sattrs2)))
                if len(merged_attrs) != len(sattrs1)+1:
                    continue
                intersecting = tuple(sorted(sattrs1.intersection(sattrs2)))

                unique1 = tuple(sattrs1.difference(sattrs2))[0]
                unique2 = tuple(sattrs2.difference(sattrs1))[0]
                idx1 = merged_attrs.index(unique1)
                idx2 = merged_attrs.index(unique2)

                diff = time.time() - self.mr.start
                if diff >= self.mr.max_wait:
                  _logger.debug("wait %d > %d exceeded" % (diff, self.mr.max_wait))
                  return

                yield (merged_attrs, self.fulljoin(intersecting, merged_attrs, idx1, idx2, pgroup1, pgroup2))

    def fulljoin(self, inter, union, idx1, idx2, groups1, groups2):
        def make_key(group):
            ret = [group.get(k, None) for k in union]
            ret[idx1] = ret[idx2] = None
            return tuple(ret)

        matches1 = defaultdict(list)
        matches2 = defaultdict(list)
        try:
            for g1 in groups1:
                matches1[make_key(g1)].append(g1[union[idx1]])
            for g2 in groups2:
                matches2[make_key(g2)].append(g2[union[idx2]])
        except:
            pdb.set_trace()

        seen = set()

        for key in matches1.keys():
            if key not in matches2:
                continue

            for v1 in matches1[key]:
                for v2 in matches2[key]:
                    newg = list(key)
                    newg[idx1] = v1
                    newg[idx2] = v2
                    newg = tuple(newg)
                    if newg in seen:
                        continue
                    seen.add(newg)
        return seen

             
