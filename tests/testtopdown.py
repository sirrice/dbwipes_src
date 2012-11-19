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
from sqlalchemy import *

from db import *
from score import *
from aggerror import *
from arch import *
from gentestdata import *
from util import reconcile_tables
from sigmod import *
from common import *

matplotlib.use("Agg")



def print_clusters(pp, clusters, tuples=[], title=''):
    fig = plt.figure(figsize=(15, 15))
    sub = fig.add_subplot(111)
    clusters.sort(key=lambda c: c.error)

    for cluster in clusters:
        x, y = tuple(map(list, zip(*cluster.bbox)))
        x[0] = max(0, x[0])
        x[1] = min(100, x[1])
        y[0] = max(0, y[0])
        y[1] = min(100, y[1])
        c = cm.jet(cluster.error)
        r = Rect((x[0], y[0]), x[1]-x[0], y[1]-y[0], alpha=max(0.1,cluster.error), ec=c, fill=False, lw=1.5)
        sub.add_patch(r)

    if tuples:
        xs, ys, cs = zip(*tuples)
        sub.scatter(ys, xs, c=cs, alpha=0.4, lw=0)


    sub.set_ylim(-5, 105)
    sub.set_xlim(-5, 105)
    sub.set_title(title)
    plt.savefig(pp, format='pdf')



def run(pp, dataset, bounds, **params):
    sigmoddb = create_engine('postgresql://localhost/sigmod')
    costs, rules, all_ids, table_size, learner = run_experiment(dataset, **params)
    truth = set(get_ids_in_bounds(sigmoddb, dataset, bounds))
    all_stats = [compute_stats(ids, truth,  table_size) for ids in all_ids]

    print "\n".join(map(str, learner.costs.items()))
#    for stats, rule, ids in zip(all_stats, rules, all_ids):
#        print stats, '\t', str(sdrule_to_clauses(rule)[0])

    try:
        all_clusters = normalize_cluster_errors(learner.all_clusters)
        clusters = normalize_cluster_errors([c.clone() for c in learner.final_clusters])
        best_clusters = sorted(clusters, key=lambda c: c.error, reverse=True)[:1]

        tuples = get_tuples_in_bounds(sigmoddb, dataset, bounds, 'g >= 7 ')
        cols = zip(*tuples)
        tuples = zip(cols[1], cols[2], [v / 100. for v in cols[-1]])

        #print_clusters(pp, all_clusters, tuples=tuples, title="all clusters")
        print_clusters(pp, clusters,  tuples=tuples,title="merged clusters %.4f" % params['c'])
        print_clusters(pp, best_clusters[:10],  tuples=tuples, title="best clusters %.4f" % params['c'])
    except:
        pass




       

if __name__ == '__main__':
    np.seterr(all='raise')
    nbadresults = 10
    idxs = sys.argv[1:] or [0,1]
    bounds = [[42.2210925762524, 92.2210925762524], [37.89772014701512, 87.89772014701512]] 
    pp = PdfPages('figs/topdown_all.pdf')
    for dataset in idxs:
        for c in reversed([0., 0.05, 0.1, 0.2, 0.3, 0.5]):
            run(pp, dataset, bounds,
                  klass=BDT, 
                  nbadresults = nbadresults,
                  epsilon=0.0001,
                  tau=[0.1, 0.75],
                  p = 0.8,
                  l=.5,
                  min_pts = 3,
                  min_improvement=.01,
                  granularity=10,
                  max_wait=60,#None,
                  naive=False,#True,
                  use_mtuples=False,
                  tablename=dataset, 
                  use_cache=True,
                  cs=[0., 0.1, 0.2],
                  c=c)

    pp.close()

#
#        cluster = None
#        while True:
#            print "set cluster to a value"
#            pdb.set_trace()
#            if not cluster:
#                break
#            pp = PdfPages('figs/topdown_%s.pdf' % outname)
#            topdown.merger.adj_matrix.insert(cluster)
#            neighbors = topdown.merger.adj_matrix.neighbors(cluster)
#            for n in neighbors:
#                n.error = 0.5
#            cluster.error = 1
#            print_clusters(pp, list(neighbors) + [cluster], title='foo')
#
#            pp.close()
#
#


