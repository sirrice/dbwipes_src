import sys
import random
import time
import sys
import matplotlib
import numpy as np
sys.path.extend( ['.', '..', '../..'])

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle as Rect
from matplotlib import cm 
import matplotlib.pyplot as plt
from collections import defaultdict

from db import *
from aggerror import *
from arch import *
from gentestdata import *
from util import reconcile_tables
from sigmod import *

matplotlib.use("Agg")




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

    return accuracy, precision, recall, r.quality, r

def kname(klass):
    return klass.__name__[-7:]



def strip_columns(table, aggerr, cols=None):
    cols = cols or [attr.name for attr in table.domain]
    cols = [col for col in cols 
            if (col not in ['id', 'err', 'epochid', 'date'] and 
                col not in aggerr.agg.cols)]
    all_cols = cols + aggerr.agg.cols
    torm = [attr.name for attr in table.domain if attr.name not in all_cols]
    table = rm_attr_from_domain(table, torm)
    return table, cols



def get_parameters(datasetidx, **kwargs):
    name = datasetnames[datasetidx]
    outname = name

    test_data = get_test_data(name)
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
    user_cols = kwargs.get('cols', None)
    table, cols = strip_columns(table, aggerr, cols=user_cols)
    bad_tables = [strip_columns(t, aggerr, cols=user_cols)[0] for t in bad_tables]
    good_tables = [strip_columns(t, aggerr, cols=user_cols)[0] for t in good_tables]
    table = full_table

    truth = set(get_ground_truth(full_table))

    return full_table, bad_tables, good_tables, truth, aggerr, cols
 

def get_clusters(full_table, bad_tables, good_tables, **kwargs):
    klass = kwargs['klass']

    learner = klass(**kwargs)
    start = time.time()
    clusters = learner(full_table, bad_tables, good_tables, **kwargs)
    end = time.time()
    cost = end - start
    costs = learner.costs
    clusters = filter(bool, clusters)
    all_clusters = learner.all_clusters


    normalize = lambda arg: normalize_cluster_errors([c.clone() for c in arg])
    clusters, all_clusters = map(normalize, (clusters, all_clusters))
    best_clusters = filter_top_clusters(clusters, nstds=1)

    cols = kwargs['cols']
    best = list(clusters_to_rules(best_clusters, cols, full_table))
    merged = list(clusters_to_rules(clusters, cols, full_table))
    allr = list(clusters_to_rules(all_clusters, cols, full_table))

    simplify = lambda rules: map(lambda r: r.simplify(), rules)
    best, merged, allr = map(simplify, (best, merged, allr)) 
    return cost, costs, best, merged, allr

def run_experiment(datasetidx, **kwargs):
    print kwargs
    ft,bts,gts, truth, aggerr, cols = get_parameters(datasetidx, **kwargs)

    params = {
            'epsilon':0.001,
            'tau':[0.01, 0.25],
            'lamb':0.5,
            'min_pts':3,
            'c' : 0.  }
    params.update(kwargs)
    params.update({
        'aggerr' : aggerr,
        'cols' : cols
        })

    cost, costs, best, merged, allr = get_clusters(ft, bts, gts, **params)

    f_stats = lambda r: compute_stats(r, truth, ft)

    bstats = map(f_stats, best)
    mstats = map(f_stats, merged)
    astats = map(f_stats, allr)
    return cost, costs, bstats, mstats, astats





    print "\n======Final Results====="
    print "Ideal: %d tuples" % len(get_ground_truth(full_table))
    for r in clusters_to_rules(best_clusters[:10], cols, table):
        r = r.simplify()

        print '%.4f\t%d\t%s' % (r.quality, len(r.examples), sdrule_to_clauses(r)[0])
        acc, pre, rec = compute_stats(r, set(get_ground_truth(full_table)), full_table)
        print acc, pre, rec


    try:
        print_clusters(pp, all_clusters, title="all clusters")
        print_clusters(pp, best_clusters, title="best clusters")
    except:
        pass


