import random
import sys
sys.path.extend( ['.', '..', '../learners/cn2sd/'])
from db import *
from parallel import *
from score import *
from aggerror import *
from arch import *
from datetime import datetime
from refiner import *
import pickle
import time
from learners.cn2sd.evaluator import *
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt



sql = '''select avg(temp),
((extract(epoch from date+time - '2004-3-1'::timestamp)) / (30*60))::int as dist
from newreadings
where date+time > '2004-3-1'::timestamp and date+time < '2004-3-7'::timestamp
group by dist order by dist;'''

badresults = [64, 72, 126, 127, 65]#, 66, 67, 68, 74, 75, 76, 122, 121, 124, 123, 125, 128, 129, 130, 131]
goodresults = [0, 1, 9, 10, 15, 20]
ignore_attrs = ['temp', 'voltage', 'humidity']#, 'light', 'epochid']
form = {'data' : json.dumps({'avg(temp)' : badresults}),
        'goodkeys' : json.dumps({'avg(temp)' : goodresults}),
        'errtype' : 2,
        'erreq' : None,
        'query' : sql,
        'bad_tuple_ids' : json.dumps({})}





def confidence_refiner(nsamples=10):
    def f(rule, err_func):
        if not len(rule.examples):
            return 0., 0., 0., 0.
        
        scores = []
        for new_rule in ContinuousBeamRefiner()(rule, ignore_attrs=ignore_attrs):
            def tmp(t, negate=True):
                return rule.filter(t, negate=negate)
            if len(new_rule.examples):
                score = err_func(tmp) / len(new_rule.examples)
                scores.append( score )
            if len(scores) > nsamples:
                break


        mean, std = np.mean(scores), np.std(scores)
        conf = mean / (abs(mean) + std)
        return mean, std, max(scores)-min(scores), conf
    return f


def confidence_sample(perc=0.1, nsamples=10):
    def f(rule, err_func):
        if not len(rule.examples):
            return 0., 0., 0., 0.

        if len(rule.examples) == 1:
            def tmp(t, negate=True):
                return t
            score = err_func(tmp)
            return score, 0., 0., 1.
        

        sampler = Orange.data.sample.SubsetIndices2(p0=perc)
        sampler.random_generator = Orange.misc.Random(len(rule.examples)) 

        scores = []
        examples = rule.examples
        for i in xrange(nsamples):
            idxs = sampler(examples)
            if not (len(idxs) - sum(idxs)):continue
            def tmp(table, negate=True):
                if not negate:
                    ret = examples.select_ref(idxs, negate=True)
                    return ret
                vals = [ row['id'] for row in examples.select_ref(idxs, negate=True)]
                predicate = ids_filter(examples, vals, attrname='id', negate=False)
                return predicate(table, negate=negate)
            score = err_func(tmp) / (len(idxs) - sum(idxs))
            scores.append( score )
        if not len(scores):
            return 0., 0., 0., 0.
        mean, std = np.mean(scores), np.std(scores)
        conf = mean / (abs(mean) + std)
        return mean, std, max(scores)-min(scores), conf
    return f



def get_rules(table):
    rules = []
    baserule = SDRule(table, '1')
    rules.append(baserule)
    for rule in ContinuousBeamRefiner()(baserule):
        if len(rule.examples):
            rules.append( rule )
            if len( [1 for row in rule.examples if row['temp'].value > 100] ):
                for nextrule in ContinuousBeamRefiner()(rule, ignore_attrs=['temp', 'id', 'epochid']):
                    if len(nextrule.examples):
                        rules.append(nextrule)
    for i in xrange(16, 25, 3):
        rules.append(baserule.cloneAndAddCondition(table.domain['moteid'], map(str, range(15, i))))
    return rules



def get_err_func(table):
    aggerr = AggErr(SelectAgg('avg', 'avg', ['temp'], 'temp', None),
                    [], 0, ErrTypes.TOOHIGH)
    err_func = FastAvgErrFunc(aggerr)
    err_func.setup(table)
    return err_func


def plot_fig(table, confidence, rules, fname, isgoodtable):
    bads = [row['temp'].value for row in table]
    bad_ids = [row['id'].value for row in table if row['temp'].value > 50]
    m = np.mean(bads)
    t = sum(bads)
    n = len(bads)
    m = t / len(table)

    err_func = get_err_func(table)

    
    xs, ys, confs = [], [], []
    for rule in rules:
        def tmp(t, negate=True):
            return rule.filter(t, negate=negate)
        score = err_func(tmp) / len(rule.examples)
        
        col = [row['temp'].value for row in rule.examples]
        x = len([c for c in col if c > 50]) / float(len(col))
        if isgoodtable:
            xs.append(len(xs))
        else:
            xs.append(x)            

        mean, std, diff, conf = confidence(rule, err_func)
        nmean, nstd, ndiff, nconf = confidence(rule.cloneAndNegate(), err_func)
        ys.append(( mean, std, diff, score, nmean, nstd, rule))
        confs.append(conf)
        sys.stdout.write('.')
        sys.stdout.flush()
    sys.stdout.write('\n')
    
    labels = ['mean', 'std', 'max-min', 'mean-score', 'negated-mean', 'negated-std']
    colors = ['blue', 'green', 'red', 'orange', 'pink', 'black', 'grey']
    pts = zip(xs, ys, confs)
    pts.sort(key=lambda p: p[0])
    xs, ys, confs = zip(*pts)

    for x, (mean, std, diff, meanscore, nmean, nstd, rule) in zip(xs, ys):
        print '\t%.4f\t%.4f\t%.6f\t%.4f\t%.4f\t%.4f\t%d\t%s' % (x, mean, std, meanscore, nmean, nstd, len(rule.examples), rule)
    print

    fig = plt.figure(figsize=(20, 15))
    sub = fig.add_subplot(111)
    sub2 = sub.twinx()
    for y, label, color in zip(zip(*ys)[:4], labels[:4], colors[:4]):
        #sub.scatter(xs, y, label=label, alpha=0.6, c=color, s=10, lw=0)
        sub.plot(xs, y, label=label, alpha=0.6, c=color)
    for y, label, color in zip(zip(*ys)[4:-1], labels[4:], colors[4:]):
        #sub.scatter(xs, y, label=label, alpha=0.6, c=color, s=10, lw=0)
        sub2.plot(xs, y, label=label, alpha=0.6, c=color)

    sub.set_xlim(-0.1, 1.1)
    sub2.set_xlim(-0.1, 1.1)
    sub.legend(loc='upper center')
    sub.set_title(fname)
    plt.savefig(pp, format='pdf')


pp = PdfPages('figs/confidencescores.pdf')
db = connect('intel')
obj = parse_debug_args(db, form)
tables = [('bad', get_provenance(obj, ['temp'], badresults))]
goodtable = get_provenance(obj, ['temp'], goodresults)
db.close()

# good_err_func = get_err_func(goodtable)
# means = []
# for rule in get_rules(goodtable):
#     mean, std, diff, conf = confidence_sample(0.01, nsamples=50)(rule, good_err_func)
#     means.append(mean)

# print np.mean(means), np.std(means), np.mean(means) + 2.5*np.std(means)

# means = []
# for rule in get_rules(goodtable):
#     mean, std, diff, conf = confidence_sample(0.01, nsamples=50)(rule.cloneAndNegate(), good_err_func)
#     means.append(mean)
# print np.mean(means), np.std(means), np.mean(means) + 2.5*np.std(means)    
# exit()


confidences = [('ref:10',confidence_refiner(10)),
               ('ref:100',confidence_refiner(100)),
               ('samp:10 0.1', confidence_sample(perc=0.01, nsamples=10)),
               ('samp:10 0.3', confidence_sample(perc=0.2, nsamples=10)),
               ('samp:100 0.1', confidence_sample(perc=0.01, nsamples=100)),
               ('samp:100 0.3', confidence_sample(perc=0.2, nsamples=100))]
confidences = confidences[2:4]

for tabletype, table in tables:
    #for row in table:
        #if row['temp'].value > 100:
        #    row['temp'] = 51
    
    rules = get_rules(table)
    for conftype, confidence in confidences:
        fname = '%s - %s' % (tabletype, conftype)
        plot_fig(table, confidence, rules, fname, tabletype == 'good')
pp.close()
