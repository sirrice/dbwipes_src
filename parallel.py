import sys
import time
import pdb
import traceback
import errfunc

from itertools import chain
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process, Queue, Pool
from Queue import Empty

#from bottomup.cluster import *
#from bottomup.bottomup import *
#from hybrid.hybrid import *
#from topdown.topdown import *
#from bottomup.svm import *


from db import *
from aggerror import *
from arch import *
from util import get_logger, rm_attr_from_domain, reconcile_tables
from sigmod import *



_logger = get_logger()


def parallel_debug(sharedobj, **kwargs):
    for aggerr in sharedobj.errors:
        parallel_runner(sharedobj, aggerr, **kwargs)
    create_clauses(sharedobj)
    

def parallel_runner(sharedobj, aggerr, **kwargs):
    sql = sharedobj.sql
    badresults = aggerr.keys
    goodresults = sharedobj.goodkeys.get(aggerr.agg.shortname, None)
    label = aggerr.agg.shortname
    if not badresults:
        return

    bad_tuple_ids = set()
    for col in aggerr.agg.cols:
        bad_tuple_ids.update( sharedobj.get_bad_tuple_ids(col) )

    if False and bad_tuple_ids:
        
        table = get_provenance(sharedobj, aggerr.agg.cols, aggerr.keys)
        scores = []
        for row in table:
            row[ERROR_VAR] = '1'            
            if row['id'].value in bad_tuple_ids:
                scores.append( 1 )
            else:
                scores.append( 0 )
        _logger.debug("Classifying using user specified tuple ids")
        return table, []


    else:
        kwargs['ignore_attrs'] = kwargs.get('ignore_attrs', sharedobj.ignore_attrs)
        #cost, ncalls, table, rules = parallel_hybrid(sharedobj, aggerr, **kwargs)
        cost, ncalls, table, rules = serial_hybrid(sharedobj, aggerr, **kwargs)

    sharedobj.merged_tables[label] = table
    rules = zip(rules, reversed(range(len(rules))))
    sharedobj.rules[label] = rules
    return table, rules



def serial_hybrid(obj, aggerr, **kwargs):
    db = connect(obj.dbname)
    obj.db = db
    bad_tables = get_provenance_split(obj, aggerr.agg.cols, aggerr.keys)
    good_tables = get_provenance_split(obj, aggerr.agg.cols, obj.goodkeys[aggerr.agg.shortname]) or []

    
    cost, ncalls = 0, 0
    rules = []
    try:
        cols = [attr.name for attr in bad_tables[0].domain
                if (attr.name not in ['id', 'err']
                    and attr.name not in aggerr.agg.cols
                    and attr.name not in kwargs['ignore_attrs'])]
        all_cols = cols + aggerr.agg.cols        
        torm = [attr.name for attr in bad_tables[0].domain if attr.name not in all_cols]

        bad_tables = [rm_attr_from_domain(t, torm) for t in bad_tables]
        good_tables = [rm_attr_from_domain(t, torm) for t in good_tables]
        (bad_tables, good_tables), full_table = reconcile_tables(bad_tables, good_tables)
        _, full_table = reconcile_tables(bad_tables)


        params = dict(kwargs)
        params.update({
            'aggerr':aggerr,
            'cols':cols})
            # errperc=0.001,
            # 
            # msethreshold=0.01,
            # k=10,
            # nprocesses=4,
            # parallelize=True,
            # complexity_multiplier=1.5)


        if aggerr.agg.func in (errfunc.SumErrFunc, errfunc.CountErrFunc):
            klass = BDT
        else:
            klass = MR


        start = time.time()
        hybrid = klass(**params)
        clusters = hybrid(full_table, bad_tables, good_tables)
        clusters = filter(lambda c: c.error >= 0, clusters)
        normalize_cluster_errors(clusters)
        clusters.sort(key=lambda c: c.error, reverse=True)
        rules = clusters_to_rules(clusters, cols, full_table)
        


        cost = time.time() - start
        ncalls = 0
    except:
        traceback.print_exc()

    
    rules.sort(key=lambda r: r.quality, reverse=True)

    print "found rules"
    print '\n'.join(map(str, rules[:5]))
    
    return cost, ncalls, table, rules




def parallel_hybrid(obj, aggerr, **kwargs):

    def f(table, aggerr, queue):
        try:

            cols = [attr.name for attr in table.domain
                    if (attr.name not in ['id', 'err']
                        and attr.name not in aggerr.agg.cols
                        and attr.name not in kwargs['ignore_attrs'])]
            print cols
            all_cols = cols + aggerr.agg.cols        
            torm = [attr.name for attr in table.domain if attr.name not in all_cols]
            table = rm_attr_from_domain(table, torm)
            
            start = time.time()
            hybrid = SVMBottomUp(aggerr=aggerr,
                            errperc=0.001,
                            cols=cols,
                            msethreshold=0.01,
                            k=10,
                            nprocesses=4,
                            parallelize=True)
            clusters = hybrid(table)
            normalize_cluster_errors(clusters)
            rules = clusters_to_rules(clusters, cols, table)
            cost = time.time() - start
            ncalls = 0
            
            queue.put( (rules, cost, ncalls) )
        except:
            traceback.print_exc()
            queue.put(None)

    nprocesses = kwargs.get('nprocesses', 4)
    parallelize = kwargs.get('parallelize', True)


    db = connect(obj.dbname)
    badresults = aggerr.keys
    queue = Queue()
    npending = 0    
    totalcost, totalncalls = 0., 0.
    partitions = parallel_get_partitions(obj, aggerr, nprocesses)
    mastertable = merge_tables(partitions)
    
    for partition in partitions:
        if parallelize:
            p = Process(target=f, args=(partition, aggerr, queue))
            p.start()
        else:
            f(partition, aggerr, queue)
        npending += 1
            

    results = []
    start = time.time()
    while npending > 0:
        try:
            result = queue.get(timeout=1)
            npending -= 1            
            if result:
                rules, cost, ncalls = result
                totalncalls += ncalls
                results.extend(rules)
            else:
                print "got error"

        except:
            pass

    results.sort(key=lambda r: r.quality, reverse=True)

    totalcost = time.time() - start
    db.close()


    results, merge_cost = parallel_rank_rules(aggerr, partitions, results, **kwargs)

    print "found rules"
    print '\n'.join(map(str, results))
    
    return totalcost, totalncalls, mastertable, results



def normalize_rules(rules):
    qualities = [r.quality for r in rules if r.quality is not None]
    if not len(qualities):
        for r in rules:
            r.quality = 0
        return rules

    minv, maxv = min(qualities), max(qualities)
    div = 1. if maxv == minv else maxv-minv
    for r in rules:
        if r.quality is None:
            r.quality = 0
        else:
            r.quality = (r.quality - minv) / div
    return rules




def parallel_rank_rules(aggerr, tables, rules, **kwargs):
    """
    Parallel rank rules based on xval
    """
    def f(aggerr, table, rules, queue):
        try:
            err_func = aggerr.error_func
            err_func.setup(table)

            cloned_rules = []
            for rule in rules:
                rule = rule.cloneWithNewData(table)
                rule.set_data(table)

                if not len(rule.examples):
                    quality = None
                else:
                    quality = err_func(rule.examples.to_numpyMA('a')[0]) / len(rule.examples)
                rule.quality = quality
                cloned_rules.append(rule)


            normalize_rules(cloned_rules)
            scores = [rule.quality or 0. for rule in cloned_rules]
            queue.put(scores)
        except:
            traceback.print_exc()
            queue.put( None )


    parallelize = kwargs.get('parallelize', True)
    rules = list(rules)
    
    queue = Queue()
    npending = 0

    for table in tables:
        if parallelize:
            Process(target=f, args=(aggerr, table, rules, queue)).start()
        else:
            f(aggerr, table, rules, queue)
        npending += 1


    
    start = time.time()
    rule_scores = defaultdict(list)
    while npending > 0:
        try:
            result = queue.get(timeout=1)
            npending -= 1
            _logger.debug("parallel_process_results\t%d processes left", npending)
            if not result:
                continue

            for rule, score in zip(rules, result):
                rule_scores[rule].append(score)

        except Empty:
            continue
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()

    for rule, scores in rule_scores.iteritems():
        rule.quality = np.mean(scores)
        
    rules = rule_scores.keys()
    rules.sort(key=lambda r: r.quality, reverse=True)
    rules = remove_subsumed_rules(rules)

    cost = time.time() - start
    return rules, cost
        








class ProvenanceGetter(object):
    def __init__(self, obj, cols):
        self.obj = obj
        self.cols = cols

    def __call__(self, keysubrange):
        obj = self.obj.clone()
        try:
            db = connect(obj.dbname)
            obj.db = db
            return get_provenance(obj, self.cols, keysubrange)
        except:
            traceback.print_exc()
        finally:
            try:
                db.close()
            except:
                pass


def parallel_get_partitions(obj, aggerr, nprocesses):
    pool = Pool(8)
    queue = Queue()
    processes = []
    args = [ ]
    chunksize = max(1, len(aggerr.keys) / nprocesses)
    partitions = pool.map(ProvenanceGetter(obj, aggerr.agg.cols), block_iter(aggerr.keys, nprocesses))
    pool.close()
    partitions = filter(lambda x:x, partitions)
    return partitions



def merge_tables(tables):
    discretes = defaultdict(list)
    conts = set()
    for table in tables:
        for attr in table.domain:
            if attr.varType == Orange.feature.Type.Discrete:
                discretes[attr.name].extend(attr.values)
            else:
                conts.add(attr.name)

    features = []
    for attr in tables[0].domain:
        if attr.name in discretes:
            features.append(Orange.feature.Discrete(attr.name, values=discretes[attr.name]))
        else:
            features.append(Orange.feature.Continuous(attr.name))
    
    domain = Orange.data.Domain(features)
    domain.add_metas(tables[0].domain.get_metas())
    ret = Orange.data.Table(domain)
    for table in tables:
        ret.extend(table)
    return ret





if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

