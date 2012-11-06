import sys
import pickle
import time
import matplotlib
sys.path.extend( ['.', '..'])

from pymongo import *
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

from db import *
from parallel import *
from score import *
from aggerror import *
from arch import *
from gentestdata import *
from learners.cn2sd.evaluator import *

matplotlib.use("Agg")


def get_roc(r, table, bad_tuple_ids):
    found_ids = set([row['id'].value for row in r.examples])
    n = len(found_ids)        
    tp = len(bad_tuple_ids.intersection(found_ids))
    fn = len(bad_tuple_ids.difference(found_ids))
    fp = len(found_ids.difference(bad_tuple_ids))
    tn = len(table) - tp - fn - fp

    accuracy = float(tp + tn) / len(table)
    precision = float(tp) / (tp + fp) if n else 0.
    recall = float(tp) / (tp + fn) if n else 0.
    return accuracy, precision, recall

def print_rule_stats(r, table, bad_tuple_ids):
    accuracy, precision, recall = get_roc(r, table)
    score = r.quality
    print "%.3f\t%.3f\t%.3f\t%.3f\t%s" % (accuracy, precision, recall, score, r)




if __name__ == '__main__':
#    import xstats.MINE

    datasetnames = ['intel_noon',
                    'intel_mote18',
                    'intel_first_spike',
                    'intel_mass_failures',
                    'fec_obama']


    nbadresults = 4

    def run_experiment(datasetidx, nbadresults=4, max_levels=3):
        name = datasetnames[datasetidx]
        pp = PdfPages('figs/rulelineage_%s.pdf' % name[name.index('_')+1:])
        test_data = get_test_data(name,
                                  nbadresults = nbadresults)

        get_ground_truth = test_data[-1]

        obj, table = create_sharedobj(*test_data[:-1])
        bad_tuple_ids = set(get_ground_truth(table))
        aggerr = obj.errors[0]

        db = connect(obj.dbname)
        obj.db = db
        goodtable = get_provenance(obj, aggerr.agg.cols, obj.goodkeys[aggerr.agg.shortname])
        err_func = aggerr.error_func
        err_func.setup(goodtable)
        good_dist = err_func.distribution(goodtable)
        db.close()


        start = time.time()

        _, allrules, stats =  classify_error_tuples_combined(
            table,
            good_dist,
            aggerr.error_func,
            cn2sd = Level_CN2_SD,
            evaluator=RuleEvaluator_RunErr_Sample,
            beamfinder = LevelBeamFinder,
            bdiscretize=False,
            #ignore_attrs=ignore_attrs,
            max_levels=max_levels)




        allrules = sorted(list(allrules), reverse=True)
        rules = allrules[:20] #+ allrules[-15:]
        for rule in rules:
            rule.set_data(table)
        getters = [('Quality', lambda r: r.quality),
                   ('Mean', lambda r: r.stats_mean),
                   ('Std', lambda r: r.stats_std),
                   ('NegMean', lambda r: r.stats_nmean),
                   ('NegStd', lambda r: r.stats_nstd)]




        fig = plt.figure(figsize=(20, 15))
        sub = fig.add_subplot(111)

        for idx, rule in enumerate(rules):
            xs, ys = [], []
            first = True
            _rule = rule
            s = (1. - math.sqrt(float(idx) / len(rules))) * 40 + 5
            c = str(math.sqrt(float(idx)/len(rules) * 0.8))

            while rule and rule.stats_mean is not None and rule.stats_nmean is not None:
                xs.append(rule.stats_mean)
                ys.append(rule.stats_nmean)
                rule = rule.parent_rule
            if not xs:
                continue

            ystr = ('%.5f\t'*len(ys)) % tuple(ys)
            print '%s\t%s' % (ystr, str(_rule))
            sub.scatter(xs[0], ys[0], s=s, c=c, alpha=1, lw=0)
            sub.plot(xs, ys, c=c)

        sub.set_title("ROC--ish")
        plt.savefig(pp, format='pdf')


        fig = plt.figure(figsize=(20, 15))
        sub = fig.add_subplot(111)
        for idx, rule in enumerate(rules):
            xs, ys, err = [], [], []
            _rule = rule
            first = True
            s = (1. - math.sqrt(float(idx) / len(rules))) * 40 + 5
            c = str(math.sqrt(float(idx) / len(rules) * 0.8))

            while rule and rule.stats_mean is not None:
                a, p, r = get_roc(rule, table, bad_tuple_ids)
                xs.append(p)
                ys.append(rule.stats_mean)
                err.append(rule.stats_std)
                rule = rule.parent_rule                
            if not xs:
                continue
            sub.scatter(xs[0], ys[0], s=s, c=c, alpha=1, lw=0)
            sub.errorbar(xs, ys, c=c, alpha=0.6)
            #p, rule.stats_mean, s=s, c=float(idx) / len(rules), alpha=0.6, lw=0)
        sub.set_title("Precision")
        pp.savefig()


        fig = plt.figure(figsize=(20, 15))
        sub = fig.add_subplot(111)
        for idx, rule in enumerate(rules):
            xs, ys, err = [], [], []
            _rule = rule
            first = True
            s = (1. - math.sqrt(float(idx) / len(rules))) * 40 + 5
            c = str(math.sqrt(float(idx) / len(rules) * 0.8))

            while rule and rule.stats_nmean is not None:
                a, p, r = get_roc(rule, table, bad_tuple_ids)
                xs.append(r)
                ys.append(rule.stats_nmean)
                err.append(rule.stats_nstd)
                rule = rule.parent_rule                
            if not xs:
                continue
            sub.scatter(xs[0], ys[0], s=s, c=c, alpha=1, lw=0)
            sub.plot(xs, ys, c=c, alpha=0.6)
            #(r, rule.stats_nmean, s=s, c=float(idx) / len(rules), alpha=0.6, lw=0)            
        sub.set_title("Recall")
        pp.savefig()

        pp.close()

    for idx in [1]:#xrange(len(datasetnames)-1):
        run_experiment(idx, max_levels=3)




    # for title, get_y in getters:
    #     fig = plt.figure(figsize=(20, 15))
    #     sub = fig.add_subplot(111)

    #     for idx, rule in enumerate(rules):
    #         xs, ys = [], []
    #         _rule = rule
    #         while rule.parent_rule:
    #             xs.append(len(rule.examples))
    #             ys.append(get_y(rule))
    #             rule = rule.parent_rule
    #         if not xs:
    #             continue

    #         lw = 5*float(idx)/len(rules) + 1
    #         c = str(float(idx)/len(rules) * 0.8)
    #         sub.plot(xs, ys, alpha=0.6, lw=lw, c=c)
    #         sub.scatter(xs[0], ys[0], s=(len(rules)-idx/float(len(rules))) * 20 + 20, lw=0, c=c)
    #     sub.set_xscale('log')
    #     sub.set_title(title)
    #     plt.savefig(pp, format='pdf')


    # rules = filter(lambda r: r.quality <= 0.9 * allrules[0].quality , allrules)[:40]
    # for title, get_y in getters:
    #     fig = plt.figure(figsize=(20, 15))
    #     sub = fig.add_subplot(111)

    #     for idx, rule in enumerate(rules):
    #         xs, ys = [], []
    #         _rule = rule
    #         while rule.parent_rule and get_y(rule) != -1:
    #             xs.append(len(rule.examples))
    #             ys.append(get_y(rule))
    #             rule = rule.parent_rule
    #         if not xs:
    #             continue

    #         lw = 1
    #         c = str(float(idx)/len(rules) * 0.8)
    #         sub.plot(xs, ys, alpha=0.6, lw=lw, c=c)
    #         sub.scatter(xs[0], ys[0], s=(len(rules)-idx/float(len(rules))) * 20 + 20, lw=0, c=c)
    #     sub.set_xscale('log')
    #     sub.set_title(title)
    #     plt.savefig(pp, format='pdf')





    # pp.close()
    # exit()

