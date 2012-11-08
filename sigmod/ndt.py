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

from settings import *
from basic import Basic

inf = 1e10000000

class NDT(Basic):


    def __call__(self, full_table, bad_tables, good_tables, **kwargs):
        """
        table has been trimmed of extraneous columns.
        """
        self.setup_tables(full_table, bad_tables, good_tables, **kwargs)

        self.SCORE_ID = add_meta_column(
                chain(self.bad_tables, self.good_tables),
                SCORE_VAR)
        self.CLASS_ID = add_meta_column(
                chain(self.bad_tables, self.good_tables),
                "INFCLASS",
                vals=['0', '1'])


        print "computing cutoffs" 
        bad_cutoff = self.influence_cutoff(self.bad_tables, self.bad_err_funcs)
        good_cutoff = self.influence_cutoff(self.good_tables, self.good_err_funcs)
        print "cutoffs\t%f\t%f" % (bad_cutoff, good_cutoff)


        import orngTree
        print "creating training data"
        training = self.create_training(bad_cutoff, good_cutoff)
        training.domain.class_var = training.domain[self.CLASS_ID]

        print "training on %d points" % len(training)
        tree = Orange.classification.tree.C45Learner(training[:100], cf=0.001)
        rules = c45_to_clauses(training, tree.tree)

        for rule in rules:
            rule.quality = self.influence(rule)

        self.final_clusters = [Cluster.from_rule(rule, self.cols) for rule in rules]
        self.all_clusters = self.final_clusters
        return self.final_clusters
 

    def influence_cutoff(self, tables, err_funcs):
        infs = []
        for table, err_func in zip(tables, err_funcs):
            for row in table:
                inf = err_func([row])
                row[self.SCORE_ID] = inf
                infs.append(inf)
        u = np.mean(infs)
        std = np.std(infs)
        return u + std * 2

    def label_bad_tuples(self, cutoff):
        pass

    def create_training(self, bad_cutoff, good_cutoff):
        extend_bad = lambda rule, t: rule.cloneAndAddContCondition(
                t.domain[self.SCORE_ID],
                bad_cutoff,
                1e100000)
        extend_good = lambda rule, t: rule.cloneAndAddContCondition(
                t.domain[self.SCORE_ID],
                good_cutoff,
                1e100000)


        train_table = Orange.data.Table(self.bad_tables[0].domain)

        for table in self.bad_tables:
            rule = SDRule(table, None)
            bad_rule = extend_bad(rule, table)
            pos_matches = bad_rule.filter_table(table)
            neg_matches =  bad_rule.cloneAndNegate().filter_table(table)
            print "bad rule: %d\t%d\t%d\t%s" % (len(table), len(pos_matches), len(neg_matches), bad_rule)

            for row in pos_matches:
                row[self.CLASS_ID] ='1'
            for row in neg_matches:
                row[self.CLASS_ID] ='0'
            train_table.extend(pos_matches)
            train_table.extend(neg_matches)


        for table in self.good_tables:
            rule = SDRule(table, None)
            good_rule = extend_good(rule, table)
            matches = good_rule.filter_table(table)
            for row in matches:
                row[self.CLASS_ID] = '0'
            train_table.extend(matches)

        return train_table



def c45_to_clauses(table, node, clauses=None):
    clauses = clauses or []
    if not node:
        return []
    
    if node.node_type == 0: # Leaf
        quality = node.class_dist[1] 
        if int(node.leaf) == 1 and node.items > 0 and clauses is not None:
            ret = [rule_from_clauses(table, clauses)]
            for rule in ret:
                rule.quality = quality
            return ret 
        return []

    var = node.tested
    ret = []


    if node.node_type == 1: # Branch
        for branch, val in zip(node.branch, var.values):
            clause = create_clause(table, var,  val)
            clauses.append( clause )
            ret.extend( c45_to_clauses(table, branch, clauses) )
            clauses.pop()

    elif node.node_type == 2: # Cut
        for branch, comp in zip(node.branch, ['<=', '>', '<', '>=']):
            clause = create_clause(table, var,  node.cut, comp)
            clauses.append( clause )
            ret.extend( c45_to_clauses(table, branch, clauses) )
            clauses.pop()

    elif node.node_type == 3: # Subset
        for i, branch in enumerate(node.branch):
            inset = filter(lambda a:a[1]==i, enumerate(node.mapping))
            inset = [var.values[j[0]] for j in inset]
            if len(inset) == 1:
                clause = create_clause(table, var, inset[0])
            else:
                clause = create_clause(table, var, inset)
            clause.append( clause )
            ret.extend( c45_to_clauses(table, branch, clauses) )
            clauses.pop()

    ret = filter(lambda c: c, ret)
    return ret



def create_clause(table, attr, val, cmp='='):
    cmps = ['<', '<=', '>', '>=', '=']
    if attr.varType == Orange.feature.Type.Discrete:
        if not isinstance(val, (list, tuple)):
            val = [val]
        vals = [orange.Value(attr, v) for v in val]
        filt = orange.ValueFilter_discrete(
            position = table.domain.index(attr),
            values = vals)
        return filt
    else:
        # it may be a discretized continuous condition (e.g., "<= 5")
        isnumerical = False
        for c in cmps:
            try:
                if val.startswith(c):
                    val = float(val.split(c)[1])
                    cmp = c
                    isnumerical = True
                    break
            except:
                pass

        if not isnumerical:
            val = float(val)


        minv, maxv = -1e10000, 1e10000
        op = None
        if cmp == '>=':
            minv = val
        elif cmp == '>':
            minv = val
        elif cmp == '<=':
            maxv = val
        elif cmp == '<':
            maxv = val
        elif cmp == '=':
            maxv = minv = val
        else:
            raise

        return Orange.data.filter.ValueFilterContinuous(
            position=table.domain.index(attr),
            oper=orange.ValueFilter.Between,
            min=minv,
            max=maxv)


def rule_from_clauses(table, clauses):
    domain = table.domain
    pos_to_cont_cond = {}
    pos_to_disc_cond = {}
    for c in clauses:
        pos = c.position
        attr = domain[pos]
        if attr.varType == Orange.feature.Type.Discrete:
            if pos in pos_to_disc_cond:
                vals = pos_to_disc_cond[pos]
                vals = vals.intersection(set(map(int, c.values)))
            else:
                vals = set(map(int, c.values))
            pos_to_disc_cond[pos] = vals
        else:
            if pos in pos_to_cont_cond:
                minv, maxv = pos_to_cont_cond[pos]
                minv = max(minv, c.min)
                maxv = min(maxv, c.max)
            else:
                minv, maxv = c.min, c.max
            pos_to_cont_cond[pos] = (minv, maxv)

    conds = []
    for pos, (minv, maxv) in pos_to_cont_cond.iteritems():
        conds.append(Orange.data.filter.ValueFilterContinuous(
            position=pos,
            oper=orange.ValueFilter.Between,
            min=minv,
            max=maxv))
    for pos, vals in pos_to_disc_cond.iteritems():
        conds.append(Orange.data.filter.ValueFilterDiscrete(
            position=pos,
            values=[Orange.data.Value(domain[pos], v) for v in vals]))
    
    return SDRule(table, None, conds)


