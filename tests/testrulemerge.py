import Orange
import orange
import sys
sys.path.append('..')

from gentestdata import *
from parallel import remove_duplicate_rules, remove_subsumed_rules
from learners.cn2sd.rule import SDRule

def disc_rule(data, name, vals):
    domain = table.domain
    attr = domain[name]
    pos = domain.index(attr)
    ovals = [orange.Value(attr, val) for val in vals]

    return orange.ValueFilter_discrete(
        position = pos,
        values = ovals
        )

def cont_rule(data, name, minv, maxv):
    domain = table.domain
    attr = domain[name]
    pos = domain.index(attr)

    return Orange.data.filter.ValueFilterContinuous(
        position=pos,
        oper=orange.ValueFilter.Between,
        min=minv,
        max=maxv)

test_data = get_test_data(datasetnames[0],
                          nbadresults = 10)
obj, table = create_sharedobj(*test_data[:-1])

conds = [[disc_rule(table, 'moteid', [16, 17])],
         [disc_rule(table, 'moteid', [17])],
         [disc_rule(table, 'moteid', [17])],
         [disc_rule(table, 'moteid', [14, 16, 17])],
         [cont_rule(table, 'humidity', -4.42, 10.06)],
         [cont_rule(table, 'humidity', -4.42, 10.03)],
         [cont_rule(table, 'humidity', -4.42, 9.82)],
         [cont_rule(table, 'humidity', -4.42, 9.88)],
         [cont_rule(table, 'humidity', -4.42, 9.87)],
         [cont_rule(table, 'humidity', -4.42, 9.97)],
         [cont_rule(table, 'humidity', -4.42, 10.13)],
         [cont_rule(table, 'voltage', 2., 2.35),
          disc_rule(table, 'moteid', [24, 25, 26, 27, 21, 22, 16, 19, 23, 17, 14])],
         [cont_rule(table, 'voltage', 2., 2.35),
          disc_rule(table, 'moteid', [24, 25, 26, 27, 14, 22, 23, 19, 17, 21]),
          cont_rule(table, 'epochid', 56208.5, 56212.25)],
         [cont_rule(table, 'voltage', 2., 2.35),
          disc_rule(table, 'moteid', [16, 17, 19, 21, 22, 23])],
          [cont_rule(table, 'voltage', 1.99, 2.34)],
          [cont_rule(table, 'voltage', 2.32, 2.34)]
         ]

rules = [SDRule(table, None, cond, 1) for cond in conds]

print len(rules)
rules = remove_duplicate_rules(rules)
print len(rules)

for idx, r in enumerate(rules):
    r.quality = -idx

rules = remove_subsumed_rules(rules)

print len(rules)
for r in rules:
    print r



