import sys
import pickle
import time
import pdb
import matplotlib
import matplotlib.pyplot as plt
sys.path.extend( ['.', '..'])

from pymongo import *
from datetime import datetime
from multiprocessing import Process, Queue
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages


from db import *
from parallel import *
from score import *
from aggerror import *
from arch import *
from gentestdata import *
from learners.cn2sd.evaluator import *
from learners.cn2sd.normalize import *
from learners.cn2sd.refiner import *


matplotlib.use("Agg")



def print_rule_stats(r, bad_tuple_ids, table, score):
    found_ids = set([row['id'].value for row in r(table)])
    n = len(found_ids)        
    tp = len(bad_tuple_ids.intersection(found_ids))
    fn = len(bad_tuple_ids.difference(found_ids))
    fp = len(found_ids.difference(bad_tuple_ids))
    tn = len(table) - tp - fn - fp

    accuracy = float(tp + tn) / len(table)
    precision = float(tp) / (tp + fp) if n and (tp+fp) else 0.
    recall = float(tp) / (tp + fn) if n and (tp+fn) else 0.
    print "%.3f\t%.3f\t%.3f\t%.3f\t%s" % (accuracy, precision, recall, score, r)



if __name__ == '__main__':

    def run_experiment(datasetidx=0):
        nbadresults = 1
        test_data = get_test_data(datasetnames[datasetidx],
                                  nbadresults = nbadresults)

        get_ground_truth = test_data[-1]

        obj, table = create_sharedobj(*test_data[:-1])
        bad_tuple_ids = set(get_ground_truth(table))
        aggerr = obj.errors[0]
        if obj.dbname == 'fec':
            ignore_attrs = ['cmte_id', 'memo_text', 'recipient_nm', 'recipient_st',
                            'recipient_zip', 'cand_id', 'form_tp', 'recipient_city',
                            'memo_cd', 'year', 'id', 'fil', 'disb_desc']
        if obj.dbname == 'intel':
            ignore_attrs = ['temp', 'epochid','humidity', 'light']#, 'voltage']
        ignore_attrs = aggerr.agg.cols + ['id'] 
        graph = defaultdict(list)
        graph.update({None : ['moteid', 'temp', 'humidity', 'voltage', 'light']})
        #'moteid' : ['humidity', 'voltage', 'light']})
        
        print ignore_attrs
        start = time.time()

        obj.ignore_attrs=ignore_attrs
        _, rules, stats = parallel_separate(obj,
                       obj.errors[0],
                       nprocesses=min(max(8, len(aggerr.keys) / 4), nbadresults),
                       parallelize=False,
                       refiner=BeamRefiner,#, graph=graph),
                       beamfinder=BoundedBeamFinder,
                       normalizer=BoundedNormalizer,
                       beta=0.4,
                       ignore_attrs = ignore_attrs,
                       evaluator = RuleEvaluator_RunErr_Sample,
                       klass=Scorer,
                       bdiscretize=False)
        print rules
        print "acc\tprec\trec\tscore"
        for rule in rules:
            rule.set_data(table)
            print_rule_stats(rule, bad_tuple_ids, table, rule.quality)

        print '-------------'
        # score by combining the top 5 rules
        filters = [r.filter for r in rules[:5]]
        merged_rule = Orange.data.filter.Disjunction(filters=filters)
        print_rule_stats(merged_rule, bad_tuple_ids, table, 0)

  
    for datasetidx in [5]:#xrange(len(datasetnames)-1):
        run_experiment(datasetidx=datasetidx)
  
