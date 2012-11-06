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

from bottomup.svm import *
from bottomup.bottomup import *
from db import *
from score import *
from aggerror import *
from arch import *
from gentestdata import *

matplotlib.use("Agg")



def print_clusters(pp, cols, clusters, title=''):
    fig = plt.figure(figsize=(15, 15))
    sub = fig.add_subplot(111)

    threshold = min(1., 1000. / (len(clusters) + 1.))
    
    for cluster in clusters:
        x, y = zip(*cluster.bbox)
        c = cm.jet(cluster.error)
        r = Rect((x[0], y[0]), x[1]-x[0]+0.1, y[1]-y[0]+0.1, alpha=0.25, fc=c, fill=True, color=c, lw=2)
        sub.add_patch(r)

    sub.set_ylim(-5, 105)
    sub.set_xlim(-5, 105)
    sub.set_xlabel(cols[0])
    sub.set_ylabel(cols[1])
    sub.set_title(title)
    plt.savefig(pp, format='pdf')


if __name__ == '__main__':
    random.seed(0)

    def kname(klass):
        return klass.__name__[-7:]


    def run_experiment(datasetidx, nbadresults=10):
        name = datasetnames[datasetidx]
        outname = name
        print name
        pp = PdfPages('figs/svm_%s.pdf' % outname)

        test_data = get_test_data(name, nbadresults = nbadresults)
        dbname, sql, badresults, goodresults, errtype, get_ground_truth = test_data
        get_ground_truth = test_data[-1]
        obj, table = create_sharedobj(*test_data[:-1])

        cols = []
        #cols = ['x', 'y']
        #cols = ['moteid', 'voltage', 'humidity', 'light', 'xloc', 'yloc']
        cols = [attr.name for attr in table.domain
                if attr.name not in ['id', 'epochid', 'err'] and attr.name not in obj.errors[0].agg.cols]
        #cols = ['light', 'moteid']
        all_cols = cols + obj.errors[0].agg.cols
        torm = [attr.name for attr in table.domain if attr.name not in all_cols]
        table = rm_attr_from_domain(table, torm)



        for perc in [0.1]:#, 0.005, 0.01]:

            k = max(int(perc * len(table)), 2)
            title = '%d points' % k
            aggerr = obj.errors[0]
            err_func = aggerr.error_func
            err_func.setup(table)
            
            bottomup = SVMBottomUp(aggerr=aggerr,
                                        err_func = err_func,
                                        k=k,
                                        cols=cols, 
                                        parallelize=False,
                                        msethreshold=0.01,
                                        nprocesses=4,
                complexity_multiplier=1.5)
            clusters = bottomup(table)
            clusters = sorted(clusters, key=lambda c: c.error, reverse=True)
            all_clusters = bottomup.all_clusters

            clusters = [c.clone() for c in clusters]
            all_clusters = [c.clone() for c in all_clusters]

            #all_clusters = normalize_cluster_errors(all_clusters)
            #clusters = normalize_cluster_errors(clusters)


            filtered_clusters = filter_top_clusters(all_clusters)
            thresh = compute_clusters_threshold(all_clusters)
            best_clusters = filter(lambda c: c.error >= thresh, clusters)
            best_rules = clusters_to_rules(best_clusters, cols, table)
            
            for r in best_rules:
                print '%.4f\t%d\t%s' % (r.quality, len(r.examples), sdrule_to_clauses(r)[0])

            print 'costs\t', (bottomup.kd_cost,
                              bottomup.sample_cost,
                              bottomup.initclusters_cost,
                              bottomup.merge_cost)
                

            if bottomup.all_clusters and len(bottomup.all_clusters[0].centroid) == 2:
                print_clusters(pp, cols, all_clusters, title=title+' initial clusters')
                print_clusters(pp, cols, filtered_clusters, title=title+' filtered')
                print_clusters(pp, cols, clusters, title=title+' merge everything')
                print_clusters(pp, cols, best_clusters, title=title+' merged then filtered')

            
        pp.close()



    nbadresults = 50
    idxs = map(int, sys.argv[1:]) or [0,1]
    for idx in idxs:
        run_experiment(idx, nbadresults=nbadresults)
