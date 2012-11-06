import sys
import random
import time
import sys
import matplotlib
sys.path.extend( ['.', '..'])

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm 
import matplotlib.pyplot as plt
from collections import defaultdict

from db import *
from score import *
from aggerror import *
from arch import *
from gentestdata import *

matplotlib.use("Agg")



def run_scorer(obj, aggerr, table, klass, **kwargs):
    kwargs['klass'] = klass
    start = time.time()
    scores, ncalls = score_inputs(table, aggerr, **kwargs)
    cost = time.time() - start
    return scores, ncalls, cost, len(table)

def compare_scores(scores, true_scores):
    mse, wmse = 0., 0.
    n = min(len(scores), len(true_scores))
    for est_score, true_score in zip(scores, true_scores):
        mse += (est_score - true_score) ** 2
        wmse += ((est_score - true_score) * max(est_score, true_score)) ** 2
    mse /= n
    wmse /= n
    return mse, wmse


def plot_points(scores, true_scores, ylabel, xlabel, table):
    from collections import Counter, defaultdict

    
    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.hexbin(scores, true_scores, gridsize=40, cmap=cm.jet)
    sub.axis([-0.2, 1.2, -0.2, 1.2])

    sub.set_xlabel(xlabel)
    sub.set_ylabel(ylabel)


if __name__ == '__main__':


    def kname(klass):
        return klass.__name__[-7:]


    def run_experiment(datasetidx, nbadresults=10):
        name = datasetnames[datasetidx]
        print name
        pp = PdfPages('figs/scorers_%s.pdf' % name)#[name.index('_')+1:])
        test_data = get_test_data(name,
                                  nbadresults = nbadresults)
        dbname, sql, badresults, goodresults, errtype, get_ground_truth = test_data
        get_ground_truth = test_data[-1]
        obj, table = create_sharedobj(*test_data[:-1])

        klasses = [QuadScoreSample7, QuadScoreSample6, QuadScoreSample5, QuadScoreSample4, Scorer]
        idxs = range(1, len(badresults)+1, 4)


        result = defaultdict(list) 
        for klass in klasses:
            aggerr = obj.errors[0]
            scores, ncalls, cost, ninputs = run_scorer(obj, aggerr, table, klass, errprob=0.01)
            print klass, ncalls, cost, ninputs
            result[klass] = [scores, ncalls, cost, ninputs]
            result['scores'] = scores
            result['ncalls'].append(ncalls)
            result['cost'].append(cost)
            result['inputsize'].append(ninputs)


        for kidx, k1 in enumerate(klasses):
            k2 = Scorer
            name1, name2 = kname(k1), kname(k2)
            scores1, scores2 = result[k1][0], result[k2][0]
            mse, wmse = compare_scores(scores1, scores2)
            print "Errors\t%s\t%s\tMSE: %f\tWMSE: %f" % (name1, name2, mse, wmse)
            plot_points(scores1, scores2, name1, name2, table)
            plt.savefig(pp, format='pdf')
            result['MSE'].append(mse)
            result['WMSE'].append(wmse)
            result[klass].extend([mse, wmse])


        for label in ['ncalls', 'cost', 'inputsize', 'MSE', 'WMSE']:
            fig = plt.figure()
            sub = fig.add_subplot(111)
            xs = np.arange(len(klasses))
            sub.bar(xs, result[label], width=0.5)
            sub.set_xticklabels(map(kname, klasses))
            sub.set_xticks(xs + 0.5)
            sub.set_ylabel(label)
            plt.savefig(pp, format='pdf')
        pp.close()

    nbadresults = 150
    for idx in [5, 6, 7, 8, 9]:#xrange(2, len(datasetnames)):    
        run_experiment(idx, nbadresults)
    exit()



    scoress = []
    ncallss = []
    costs = []
    
    for klass in klasses:
        print "getting scores using", klass.__name__

    # compute pairwise errors
    # plot pairwise comparisons
    for k1, s1 in zip(klasses, scoress):
        for k2, s2 in zip(klasses, scoress):
            if k1 == k2: continue
            mse, wmse = compare_scores(s1, s2)
            print "Errors\t%s\t%s\tMSE: %f\tWMSE: %f" % (k1.__name__, k2.__name__, mse, wmse)
            #plot_points(s1, s2, k1.__name__, k2.__name__, table)
    pp.close()

    print "Nrows\t       \t%d" % len(table)
    for k, n in zip(klasses, ncallss):
        print "Ncalls\t%s\t%d" % (k.__name__,n)

    for k, c in zip(klasses, costs):
        print "Costs\t%s\t%d" % (k.__name__, c)
    exit()
    ignore_attrs = [table.domain[col] for col in aggerr.agg.cols]

    for klass, scores in zip(klasses, scoress):
        print "Rules using %s" % klass.__name__
        table, rules = classify_error_tuples(table, scores, ignore_attrs=ignore_attrs)
        for r in rules:
            print rule_to_clauses(r)
        
    db.close()
    exit()

    # rows = [ [1, 0, 0] , [1, 1, 1], [0, 1, 2], [0, 0, 3] ]
    # table = create_orange_table(rows, ['a1', 'a2', 'id'], [])
    # scores = [0.5, 0.5, 0.1, 0.1]
    # for row in table:
    #     row[ERROR_VAR] = '1'


    import random
    rows = [[random.random(), 0, idx] for idx in range(50)]
    rows.extend( [[100, 1, idx+100] for idx in xrange(30)] )
    
    table = create_orange_table(rows, ['target', 'attr', 'id'], [])
    scores = leave_one_out_score(table, aggerr)
    maxscore = max(scores)
    scores = map(lambda s: s/maxscore, scores)
    for row, score in zip(table, scores):
        row['err'] = score > 0.8 and '1' or '0'

    table = table.select([1,2,3])
    rules = classify_error_tuples(table, scores)

