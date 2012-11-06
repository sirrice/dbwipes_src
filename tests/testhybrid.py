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

from bottomup.cluster import *
from hybrid.hybrid import Hybrid
from db import *
from score import *
from aggerror import *
from arch import *
from gentestdata import *

matplotlib.use("Agg")



def print_clusters(pp, clusters, title=''):
    fig = plt.figure(figsize=(15, 15))
    sub = fig.add_subplot(111)

    for cluster in clusters:
        x, y = zip(*cluster.bbox)
        c = cm.jet(cluster.error)
        r = Rect((x[0], y[0]), x[1]-x[0], y[1]-y[0], alpha=0.45, fc=c, fill=True, lw=0)
        sub.add_patch(r)

    sub.set_ylim(-5, 105)
    sub.set_xlim(-5, 105)
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
        pp = PdfPages('figs/hybrid_%s.pdf' % outname)

        test_data = get_test_data(name, nbadresults = nbadresults)
        dbname, sql, badresults, goodresults, errtype, get_ground_truth = test_data
        get_ground_truth = test_data[-1]
        obj, table = create_sharedobj(*test_data[:-1])

        cols = ['x', 'y', 'v']
        cols = [attr.name for attr in table.domain
                if attr.name not in ['id', 'err'] and attr.name not in obj.errors[0].agg.cols]
        all_cols = cols + obj.errors[0].agg.cols        
        torm = [attr.name for attr in table.domain if attr.name not in all_cols]
        table = rm_attr_from_domain(table, torm)

        
        errperc = 0.001
        hybrid = Hybrid(aggerr=obj.errors[0],
                        errperc=errperc,
                        cols=cols,
                        msethreshold=0.01,
                        perc_k = 0.01,
                        nprocesses=4,
                        parallelize=False)
        clusters = hybrid(table)
        clusters.sort(key=lambda c: c.error, reverse=True)        
        all_clusters = hybrid.all_clusters

        clusters = [c.clone() for c in clusters]
        all_clusters = [c.clone() for c in all_clusters]

        all_clusters = normalize_cluster_errors(all_clusters)
        clusters = normalize_cluster_errors(clusters)

        filtered_clusters = filter_top_clusters(all_clusters)
        best_clusters = filter_top_clusters(clusters, nstds=0.5)
        print "results: %d clusters, %d best clusters out of %d" % (len(clusters),
                                                                    len(best_clusters),
                                                                    len(all_clusters))

        

        title = 'err=%.4f' % errperc

        if all_clusters:
            if len(all_clusters[0].bbox[0]) == 2:
                print_clusters(pp, all_clusters, title=title+' initial clusters')
                print_clusters(pp, filtered_clusters, title=title+' filtered')                
                print_clusters(pp, clusters, title=title+' merged clusters')
                print_clusters(pp, best_clusters, title=title+' best merged clusters')
            else:
                best_rules = clusters_to_rules(best_clusters, cols, table)
                for r in best_rules:
                    print '%.4f\t%d\t%s' % (r.quality, len(r.examples), sdrule_to_clauses(r)[0])

            
        pp.close()


    nbadresults = 5
    idxs = map(int, sys.argv[1:]) or [0,1]
    for idx in idxs:
        run_experiment(idx, nbadresults=nbadresults)
