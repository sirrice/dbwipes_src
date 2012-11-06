import sys
import pickle
import time
import math
import random
import matplotlib
import numpy as np
sys.path.extend( ['.', '..'])

from pymongo import *
from datetime import datetime
from multiprocessing import Process, Queue
from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from db import *
from parallel import *
from score import *
from aggerror import *
from arch import *
from gentestdata import *
from learners.cn2sd.evaluator import *

matplotlib.use("Agg")


def plot_correlation_matrix(data, fig=None, sub=None):
    """
    data: [ (attr1, attr2) -> value, ... ]
    """
    datadict = defaultdict(dict)
    keys = set()
    for (attr1, attr2), val in data:
        keys.update((attr1, attr2))
        datadict[attr1][attr2] = val
        datadict[attr2][attr1] = val
        datadict[attr1][attr1] = 1.
        datadict[attr2][attr2] = 1.
    keys = list(keys)
    nkeys = len(keys)
    idxs = np.arange(nkeys)

    if not sub:
        if not fig:
            fig = plt.figure()
        sub = fig.add_subplot(111)
    sub.set_xticklabels(keys)
    sub.set_xticks(idxs + 0.5)
    sub.set_yticklabels(keys)
    sub.set_yticks(idxs + 0.5)

    for idx, key in enumerate(keys):
        colors = map(str, [(datadict[key][k2] + 1.) / 2 for k2 in keys])
        print colors
        height = [1] * nkeys
        sub.bar(idxs, height=height, width=1., bottom=float(idx), color=colors, lw=1)
    return fig, sub

    

    
    
    


if __name__ == '__main__':
    import xstats.MINE
    pp = PdfPages('figs/correlation.pdf')
    datasetnames = ['intel_noon',
                    'intel_mote18',
                    'intel_first_spike',
                    'intel_mass_failures',
                    'fec_obama']

    def run_plot(idx):
        datasetname = datasetnames[idx]
        test_data = get_test_data(datasetname, nbadresults=2)
        obj, table = create_sharedobj(*test_data[:-1])



        data = []
        matrix, = table.to_numpy('a')
        for colidx in xrange(matrix.shape[1]):
            c1 = matrix[:,colidx]
            for c2idx in xrange(colidx+1, matrix.shape[1]):
                label1, label2 = table.domain[colidx].name, table.domain[c2idx].name
                if label1 == 'id' or label2 == 'id':
                    continue
                #if (table.domain[colidx].var_type == Orange.feature.Type.Discrete or
                #    table.domain[c2idx ].var_type == Orange.feature.Type.Discrete):
                #    continue


                c2 = matrix[:,c2idx]
                #mic = random.random()
                mic = xstats.MINE.analyze_pair(c1, c2)['MIC']
                #mic = pearsonr(c1, c2)[0]
                if math.isnan(mic):
                    continue
                data.append(((label1, label2), mic))
                print '%s\t%s\t%f' % (table.domain[colidx], table.domain[c2idx], mic)
                continue

                fig = plt.figure(figsize=(15, 10))
                sub = fig.add_subplot(111)
                sub.scatter(c1, c2, lw=0, s=5, alpha=0.4)
                sub.set_xlabel(label1)
                sub.set_ylabel(label2)
                sub.set_title('%s vs %s' % (label1, label2))
                plt.savefig(pp, format='pdf')


        fig = plt.figure(figsize=(15, 10))
        fig, sub = plot_correlation_matrix(data, fig=fig)
        sub.set_title(datasetname)
        plt.savefig(pp, format='pdf')


    for idx in xrange(len(datasetnames)-1):
        run_plot(idx)
    pp.close()
