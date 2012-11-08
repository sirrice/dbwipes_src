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
from collections import defaultdict

from db import *
from score import *
from aggerror import *
from arch import *
from gentestdata import *
from util import reconcile_tables
from sigmod import *

matplotlib.use("Agg")



def print_clusters(pp, clusters, title=''):
    fig = plt.figure(figsize=(15, 15))
    sub = fig.add_subplot(111)

    for cluster in clusters:
        x, y = tuple(map(list, zip(*cluster.bbox)))
        x[0] = max(0, x[0])
        x[1] = min(100, x[1])
        y[0] = max(0, y[0])
        y[1] = min(100, y[1])
        c = cm.jet(cluster.error)
        r = Rect((x[0], y[0]), x[1]-x[0], y[1]-y[0], alpha=0.45, fc=c, fill=True, lw=1)
        sub.add_patch(r)

    sub.set_ylim(-5, 105)
    sub.set_xlim(-5, 105)
    sub.set_title(title)
    plt.savefig(pp, format='pdf')




def compute_stats(r, bad_tuple_ids, table):
    all_found_ids = set()

    indiv_stats = []
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

    return accuracy, precision, recall








if __name__ == '__main__':
    random.seed(0)

    def kname(klass):
        return klass.__name__[-7:]



    def strip_columns(table, aggerr):
        cols = [attr.name for attr in table.domain
                if attr.name not in ['id', 'err', 'epochid', 'date', 'memo_text', 'disb_desc'] and attr.name not in aggerr.agg.cols]
        all_cols = cols + aggerr.agg.cols
        torm = [attr.name for attr in table.domain if attr.name not in all_cols]
        table = rm_attr_from_domain(table, torm)
        return table, cols


    def run_experiment(datasetidx, nbadresults=10):
        name = datasetnames[datasetidx]
        outname = name
        print name
        pp = PdfPages('figs/topdown_%s.pdf' % outname)

        test_data = get_test_data(name, nbadresults = nbadresults)
        dbname, sql, badresults, goodresults, errtype, get_ground_truth = test_data
        get_ground_truth = test_data[-1]
        obj, table = create_sharedobj(*test_data[:-1])
        aggerr = obj.errors[0]


        # retrieve table for each good and bad key
        obj.db = connect(dbname)
        bad_tables = get_provenance_split(obj, aggerr.agg.cols, aggerr.keys) or []
        good_tables = get_provenance_split(obj, aggerr.agg.cols, obj.goodkeys[aggerr.agg.shortname]) or []
        obj.db.close()

        (bad_tables, good_tables), full_table = reconcile_tables(bad_tables, good_tables)
        _, full_table = reconcile_tables(bad_tables)

        # strip unnecessary columns
        table, cols = strip_columns(table, aggerr)
        bad_tables = [strip_columns(t, aggerr)[0] for t in bad_tables]
        good_tables = [strip_columns(t, aggerr)[0] for t in good_tables]
        table = full_table
        print cols
        
        errperc = 0.001
        np.seterr(all='raise')
        topdown = BDT(aggerr=obj.errors[0],
                          errperc=errperc,
                          epsilon=0.001,
                          cols=cols,
                          msethreshold=.25,
                          tau=[0.01, 0.15],
                          lamb=0.5,
                          min_pts = 1,
                          complexity_multiplier=1.5,
                          c= 1.3)
        clusters = topdown(full_table, bad_tables, good_tables)
        clusters = filter(lambda x:x, clusters)

        all_clusters = normalize_cluster_errors([c.clone() for c in topdown.all_clusters])
        clusters = normalize_cluster_errors([c.clone() for c in clusters])
        best_clusters = filter_top_clusters(clusters, nstds=1)

        print "\n======Final Results====="
        print "Ideal: %d tuples" % len(get_ground_truth(full_table))
        for r in clusters_to_rules(best_clusters[:10], cols, table):
            print '%.4f\t%d\t%s' % (r.quality, len(r.examples), sdrule_to_clauses(r)[0])
            acc, pre, rec = compute_stats(r, set(get_ground_truth(full_table)), full_table)
            print acc, pre, rec


#title = 'err=%.4f' % errperc
#        if (best_clusters and
#            max(len(c.centroid) for c in best_clusters) == 2 and
#            min(len(c.centroid) for c in best_clusters) == 2):
#            print_clusters(pp, all_clusters, title=title+' initial clusters')
#            print_clusters(pp, clusters, title=title+' merged clusters')
#            print_clusters(pp, best_clusters, title=title+' best merged clusters')



            
        pp.close()


    nbadresults = 10
    idxs = map(int, sys.argv[1:]) or [0,1]
    for idx in idxs:
        run_experiment(idx, nbadresults=nbadresults)
