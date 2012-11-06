import sys
import random
import time
import sys
import matplotlib
import numpy as np
sys.path.extend( ['.', '..'])

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle as Rect
from matplotlib import cm 
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections import defaultdict

from topdown.topdown import TopDown
from hybrid.hybrid import Hybrid
from bottomup.svm import SVMBottomUp
from bottomup.bottomup import *
from bottomup.cluster import *
from learners.cn2sd.rule import SDRule
from db import *
from score import *
from aggerror import *
from arch import *
from gentestdata import *

matplotlib.use("Agg")



def plot_results(pp, name, titles, indiv_statss, union_statss, costs):
    lefts = np.arange(len(titles))
    w = 0.2


    for sidx, statname in enumerate(['acc', 'prec', 'recall']):
        fig = plt.figure(figsize=(15, 15))
        sub = fig.add_subplot(111)

        for x, stats in zip(lefts, indiv_statss):
            if not stats:
                continue

            color = [cm.jet(i / float(len(stats))) for i in xrange(len(stats))]
            ys = [stat[sidx] for stat in stats]
            sub.scatter([x] * len(ys), ys, color=color, s=150, alpha=0.6, lw=0)

        sub.set_ylim(0, 1.2)
        sub.set_xticks(lefts)
        sub.set_xticklabels(titles, rotation='vertical')
        sub.set_title('%s: %s' % (name, statname))
        plt.savefig(pp, format='pdf')

    # union_stats
    plot_union_stats(pp, name, titles, union_statss)

    plot_costs(pp, name, titles, costs)


def plot_costs(pp, name, titles, costs)
    lefts = np.arange(len(titles))
    w = 0.2

    # costs
    fig = plt.figure(figsize=(15, 15))
    sub = fig.add_subplot(111)
    sub.bar(lefts, costs, lw=0, alpha=0.6)
    sub.set_xticks(lefts + 0.3)
    sub.set_xticklabels(titles, rotation='vertical')
    sub.set_title('%s: runtime' % name)
    plt.savefig(pp, format='pdf')
    

def plot_union_stats(pp, name, titles, union_statss):
    lefts = np.arange(len(titles))
    w = 0.2
    
    fig = plt.figure(figsize=(15, 15))
    sub = fig.add_subplot(111)
    accs, precs, recalls = zip(*union_statss)
    
    sub.bar(lefts, accs, width=w, color='red', label='acc', lw=0, alpha=0.6)
    sub.bar(lefts+w, precs, width=w, color='blue', label='precision', lw=0, alpha=0.6)
    sub.bar(lefts+w+w, recalls, width=w, color='orange', label='recall', lw=0, alpha=0.6)

    sub.set_ylim(0, 1.2)
    sub.set_xticks(lefts+0.3)
    sub.set_xticklabels(titles, rotation='vertical')
    sub.set_title('union %s' % name)
    sub.legend(loc='upper center', ncol=2)    
    plt.savefig(pp, format='pdf')






def run_scorer(obj, aggerr, table, klass, **kwargs):
    kwargs['klass'] = klass
    _, rules, stats = parallel_separate(obj, aggerr)
    cost = stats[0]
    for rule in rules:
        print rule
    print "scorer costs\t%.4f" % cost        
    return cost, rules

def run_topdown(aggerr, table, **kwargs):
    cols = kwargs['cols']
    kwargs['aggerr'] = aggerr
    
    topdown = TopDown(**kwargs)
    clusters = topdown(table)

    thresh = compute_clusters_threshold(topdown.all_clusters)
    top_clusters = filter(lambda c: c.error >= thresh, clusters)
    top_clusters.sort(key=lambda c: c.error, reverse=True)
    
    rules = clusters_to_rules(top_clusters, cols, table)
    for r in rules:
        print r.quality, '\t', r

    cost = sum((topdown.scorer_cost, topdown.merge_cost))
    print "topdown costs\t%.4f" % cost
    return cost, rules

def run_hybrid(aggerr, perc, table, **kwargs):
    cols = kwargs['cols']
    kwargs['aggerr'] = aggerr
    kwargs['perc_k'] = perc

    hybrid = Hybrid(**kwargs)
    clusters = hybrid(table)

    thresh = compute_clusters_threshold(hybrid.all_clusters)
    top_clusters = filter(lambda c: c.error >= thresh, clusters)
    top_clusters.sort(key=lambda c: c.error, reverse=True)
    
    rules = clusters_to_rules(top_clusters, cols, table)
    for r in rules:
        print r.quality, '\t', r

    cost = hybrid.cost
    print "topdown costs\t%.4f" % cost
    return cost, rules


def run_svm(aggerr, perc, table, **kwargs):
    err_func = aggerr.error_func
    err_func.setup(table)

    cols = kwargs['cols']
    k = int(perc * len(table))
    kwargs['perc_k'] = perc
    kwargs['err_func'] = err_func
    kwargs['aggerr'] = aggerr

    svm = SVMBottomUp(**kwargs)
    clusters = svm(table)

    thresh = compute_clusters_threshold(svm.all_clusters)
    top_clusters = filter(lambda c: c.error >= thresh, clusters)
    top_clusters.sort(key=lambda c: c.error, reverse=True)
    
    rules = clusters_to_rules(top_clusters, cols, table)
    for r in rules:
        print r.quality, '\t', r

    cost = svm.cost
    print "topdown costs\t%.4f" % cost
    return cost, rules
    



def run_bottomup(aggerr, klass, perc, table, **kwargs):
    err_func = aggerr.error_func
    err_func.setup(table)

    cols = kwargs['cols']
    k = int(perc * len(table))
    kwargs['perc_k'] = perc
    kwargs['err_func'] = err_func

    bottomup = klass(**kwargs)
    clusters = bottomup(table)

    thresh = compute_clusters_threshold(bottomup.all_clusters)
    top_clusters = filter(lambda c: c.error >= thresh, clusters)
    top_clusters.sort(key=lambda c: c.error, reverse=True)

    rules = clusters_to_rules(top_clusters, cols, table)
    for r in rules:
        print r.quality, r
    

    cost = sum((bottomup.sample_cost,
               bottomup.initclusters_cost,
               bottomup.merge_cost))
    
    print "bottomup costs\t%s" % ('\t'.join(['%.4f']*3) % (bottomup.sample_cost,
                                                           bottomup.initclusters_cost,
                                                           bottomup.merge_cost))
    
    return cost, rules




def compute_stats(rules, bad_tuple_ids, table):
    all_found_ids = set()

    indiv_stats = []
    for r in rules:
        found_ids = set([row['id'].value for row in r(table)])
        all_found_ids.update(found_ids)
        n = len(found_ids)        
        tp = len(bad_tuple_ids.intersection(found_ids))
        fn = len(bad_tuple_ids.difference(found_ids))
        fp = len(found_ids.difference(bad_tuple_ids))
        tn = len(table) - tp - fn - fp

        accuracy = float(tp + tn) / len(table)
        precision = float(tp) / (tp + fp) if n and (tp+fp) else 0.
        recall = float(tp) / (tp + fn) if n and (tp+fn) else 0.
        indiv_stats.append((accuracy, precision, recall))


    n = len(all_found_ids)        
    tp = len(bad_tuple_ids.intersection(all_found_ids))
    fn = len(bad_tuple_ids.difference(all_found_ids))
    fp = len(all_found_ids.difference(bad_tuple_ids))
    tn = len(table) - tp - fn - fp
    union_accuracy = float(tp + tn) / len(table)    
    union_precision = float(tp) / (tp + fp) if n and (tp+fp) else 0.
    union_recall = float(tp) / (tp + fn) if n and (tp+fn) else 0.
    union_stats = (union_accuracy, union_precision, union_recall)
    
    return indiv_stats, union_stats




if __name__ == '__main__':
    random.seed(0)

    def kname(klass):
        return klass.__name__[-7:]


    def run_experiment(pp, datasetidx, nbadresults=10):
        name = datasetnames[datasetidx]
        outname = name
        local_pp = PdfPages('figs/endtoend_%s.pdf' % name)
        print name


        test_data = get_test_data(name, nbadresults = nbadresults)
        dbname, sql, badresults, goodresults, errtype, get_ground_truth = test_data
        get_ground_truth = test_data[-1]
        obj, table = create_sharedobj(*test_data[:-1])
        bad_tuple_ids = set(get_ground_truth(table))
        
        cols = ['moteid', 'voltage', 'humidity', 'light']
        cols = ['x', 'y']
        cols = [attr.name for attr in table.domain
                if attr.name not in ['id', 'err'] and attr.name not in obj.errors[0].agg.cols]
        all_cols = cols + obj.errors[0].agg.cols
        print all_cols
        torm = [attr.name for attr in table.domain if attr.name not in all_cols]
        thintable = rm_attr_from_domain(table, torm)


        for msethreshold in [0.01, 0.1]:
            titles = []
            costs = []
            ruless = []
            indiv_statss = []
            union_statss = []

            for perc in [0.1, 0.01, 0.001]:
                percstr = ('%.4f' % perc).rstrip('0.') or '0'
                titles.append('ParBU\nperc=%s' % percstr)
                cost, rules = run_bottomup(obj.errors[0], DiscreteBottomUp, perc, thintable, cols=cols,
                                           parallelize=True,
                                           nprocesses=8,
                                           msethreshold=msethreshold)

                ruless.append(rules)
                costs.append(cost)

            for perc in [0.001]:
                percstr = ('%.4f' % perc).rstrip('0.') or '0'
                titles.append('ContBU\nperc=%s' % percstr)
                cost, rules = run_bottomup(obj.errors[0], ZeroBottomUp, perc, thintable,
                                           cols=cols,
                                           msethreshold=msethreshold)
                ruless.append(rules)
                costs.append(cost)
            

            titles.append('TopDown')
            cost, rules = run_topdown(obj.errors[0], thintable,
                                      cols=cols,
                                      errperc=0.001,
                                      msethreshold=msethreshold,
                                      complexity_multiplier=2.5)
            ruless.append(rules)
            costs.append(cost)

            # titles.append('Hybrid')
            # cost, rules = run_hybrid(obj.errors[0], 0.001, thintable,
            #                          cols=cols, errprob=0.01, parallelize=False, msethreshold=0.01)
            # ruless.append(rules)
            # costs.append(cost)

            titles.append('FeatureRed')
            cost, rules = run_svm(obj.errors[0], 0.001, thintable,
                                  cols=cols, errprob=0.01, parallelize=True,
                                  nprocesses=4,
                                  msethreshold=msethreshold)
            ruless.append(rules)
            costs.append(cost)


            for rules in ruless:
                indiv_stats, union_stats = compute_stats(rules, bad_tuple_ids, table)
                indiv_statss.append(indiv_stats)
                union_statss.append(union_stats)

            title = '%s: %d points, %.4f threshold' % (name, len(table), msethreshold)
            plot_results(local_pp, name, titles, indiv_statss, union_statss, costs)
            plot_union_stats(pp, '%s %.4f threshold' % (name, msethreshold), titles, union_statss)
            plot_costs(pp, name, titles, costs)

        local_pp.close()        



    nbadresults = 50
    pp = PdfPages('figs/endtoend.pdf')
    idxs = map(int, sys.argv[1:]) or [0,1]
    for idx in idxs:
        run_experiment(pp, idx, nbadresults)
    pp.close()    
