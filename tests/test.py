import sys
import pdb
import random
sys.path.extend(('.', '..'))


from common import *
from misc.gensinglecluster import gen_points
from util import get_logger

_logger = get_logger()
random.seed(0)


    # configurations
    # ndt: nothing, c
    # basic: max complexity, c
    # bdt: epsilon, tau, lambda, c
    # mc: c
def print_stats(pp, stats, rules, title):
    fig = plt.figure(figsize=(16, 4))
    plt.suptitle(title)
    ss = [rule.quality for rule in rules]
    accs, ps, rs, f1 = zip(*stats)
    for idx, (ys, statname) in enumerate(((accs, 'acc'), (ps, 'prec'), (rs, 'recall'), (f1, 'f1'))):
        sub = fig.add_subplot(int('14'+str(idx)))
        sub.scatter(ss, ys, lw=0)

        sub.set_ylim(0, 1.2)
        sub.set_title(statname)

    plt.savefig(pp, format='pdf')


def make_params(**kwargs):
    ret = []
    for klass in kwargs.get('klasses', [NDT, BDT, MR, Naive]):
        for c in kwargs.get('c', [0, 0.5, 0.75, 1.]):

            params = dict(kwargs)
            params.update({'c':c, 'klass':klass, 'klassname' : klass.__name__[-7:], 'p' : 0.6})
            params['cols'] = kwargs.get('cols', None)

            if klass == BDT:
                for epsilon in kwargs.get('epsilon', [0.001, 0.01, 0.05]):
                    for l in kwargs.get('l', [0.5, 1.]):
                        params2 = dict(params)
                        params2.update({ 
                            'epsilon':epsilon, 
                            'l':l, 
                            'min_improvement' : .01})
                        ret.append(params2)
            elif klass == Naive:
                for l in kwargs.get('l', [0.5]):
                    params2 = dict(params)
                    params2['max_complexity'] = kwargs.get('max_complexity',4)
                    params2['granularity'] = kwargs.get('granularity', 20) 
                    params2['l'] = l
                    ret.append(params2)
            else:
                for l in kwargs.get('l', [0.5, 1.]):
                    params['l'] = l
                    ret.append(params)

    ret.sort(key=lambda p: (p['klassname'], p['c']))
    return ret

def mkfmt(arr):
    mapping = [(float, '%.4f'), (int, '%d'),  (object, '%s')]
    fmt = []
    for v in arr:
        for t,f in mapping:
            if isinstance(v,t):
                fmt.append(f)
                break

    return '\t'.join(fmt)


def get_ground_truth(db, datasetidx, c):
    with db.begin() as conn:
        q = """select ids from stats where klass = 'Naive' and dataset = %s and c = %s order by expid desc"""
        for row in conn.execute(q, datasetidx, c).fetchall():
            return map(int, map(float, row[0].split(',')))
        return None



def run(db, datasetidx, pp=None, **kwargs):
    expid = nextexpid(db)
    pkeys = ['klassname', 'cols', 'epsilon', 'c', 'lambda']
    for params in make_params(**kwargs):
        try:
            run_id = str(params)
            costs, rules, all_ids, table_size, learner = run_experiment(datasetidx, **params)
            total_cost = costs['cost_total']
            if params['klassname'] == 'Naive':
                rules = rules[:1]
                all_ids = all_ids[:1]
                truth = all_ids[0]
            else:
                truth = get_ground_truth(db, datasetidx, params['c'])

            if truth == None:
                pdb.set_trace()
                raise RuntimeError("Could not find ground truth for %d, %.4f" % (datasetidx, params['c']))

            all_stats = [compute_stats(ids, truth, table_size) for ids in all_ids]

            row_id = None
            for (acc, p, r, f1), rule, ids in zip(all_stats, rules, all_ids):
                ids_str = ','.join(map(str, ids))
                isbest = rule.isbest
                vals = [expid, str(datasetidx), kwargs.get('notes', '')]
                vals.extend([params.get(key, None) for key in pkeys])
                vals.extend((total_cost, acc, p, r, f1, rule.quality, isbest, str(rule), ids_str))
                row_id = save_result(db, vals, costs)
                print mkfmt(vals[:-1]) % tuple(vals[:-1])

            if params['klassname'] == 'Naive':
                rules = rules[:1]
                all_ids = all_ids[:1]
                truth = all_ids[0]
                for timein, rule in learner.checkpoints:
                    chk_ids = get_row_ids(rule, learner.full_table)
                    acc, p, r, f1 = compute_stats(chk_ids, truth, table_size)
                    ids_str = ','.join(map(str, map(int, chk_ids)))
                    vals = [expid, datasetidx, kwargs.get('notes', '')]
                    vals.extend([params.get(key, None) for key in pkeys])
                    vals.extend((timein, acc, p, r, f1, rule.quality, False, str(rule), True, ids_str))
                    row_id = save_checkpoint(db, vals)
                    print mkfmt(vals[:-1]) % tuple(vals[:-1])




            if pp:
                pass
                #print_stats(pp, all_stats, rules, ','.join(map(lambda p: '%s:%s'%tuple(p), params.items())))
        except Exception as e:
            import traceback
            traceback.print_exc()
            _logger.error(traceback.format_exc())
    try:
        complete(db, expid)
    except:
        import traceback
        traceback.print_exc()


def save_result(db, stat, costs):
    with db.begin() as conn:
        q = """insert into stats(expid, dataset, notes, klass, cols, epsilon, c,     
                                 lambda, cost, acc, prec, recall, f1, score, 
                                 isbest, rule, ids) values(%s) returning id""" % (','.join(['%s']*len(stat)))
        sid = conn.execute(q, *stat).fetchone()[0]

        q = """insert into costs(sid, name, cost) values(%s,%s,%s)"""
        for name, cost in costs.items():
            if isinstance(cost, list):
                cost = cost[0]
            conn.execute(q, sid, name, cost)
        return sid

def save_checkpoint(db, stat):
    with db.begin() as conn:
        q = """insert into stats(expid, dataset, notes, klass, cols, epsilon, c,     
                                 lambda, cost, acc, prec, recall, f1, score, 
                                 isbest, rule, ischeckpoint, ids) values(%s) returning id""" % (','.join(['%s']*len(stat)))
        sid = conn.execute(q, *stat).fetchone()[0]
        return sid



def complete(db, expid):
    with db.begin() as conn:
        q = """update stats set completed = TRUE where expid = %s"""
        conn.execute(q, expid)

def nextexpid(db):
    with db.begin() as conn:
        q = """select max(expid)+1 from stats"""
        row = conn.execute(q).fetchone()
        expid = row[0]
        if expid == None:
            return 0
        return expid



def init_db(db):
    try:
        with db.begin() as conn:
            conn.execute("""create table stats (
        expid int,
        id serial,
        tstamp timestamp default current_timestamp,
        dataset text,
        klass varchar(128) null,
        cols text null,
        epsilon float null,
        c float null,
        lambda float null,
        cost float,
        acc float,
        prec float,
        recall float,
        f1 float,
        score float,
        isbest bool,
        ischeckpoint bool default FALSE,
        checkpoint float null,
        completed bool,
        rule text null,
        ids text null,
        notes text null,
        boundtype text null
                )""")
            conn.execute("""create table costs (
            id serial,
            sid int,
            name varchar(128),
            cost float)""")
    except:
        pass

def load_data(tablename, schema, pts):
    with file('/tmp/clusters.txt', 'w') as f:
        print >>f, '\t'.join(schema)
        for pt in pts:
            print >>f, '\t'.join(['%.4f']*len(pt)) % tuple(pt)
    os.system('importmydata.py %s sigmod /tmp/clusters.txt 2> /tmp/dbtruck.err > /tmp/dbtruck.log' % tablename)


def run_sigmod(pp, ndim, volperc):
    params = []
    for kdim in xrange(1, ndim+1):
        for uo in (30, 80):
            tablename = 'data_%d_%d_1000_0d%d_%duo' % (ndim, kdim, int(100.*volperc), uo)
            params.append((tablename, (2000, ndim, kdim, volperc, 10, 10, uo, 10)))

    for tablename, args in params:

        # generate the damn data
        obox, sbox, schema, pts = gen_points(*args)
        print obox
        print sbox
        print tablename
        load_data(tablename, schema, pts)
        if ndim == 2 and kdim == 1:
            continue

        cs = [0., .25, .5, .75, 1.]
        run(db, tablename, pp, klasses=[Naive], max_wait=2*60, c=cs, granularity=20, naive=True, notes=tablename)
        run(db, tablename, pp, klasses=[NDT],  c=cs, granularity=20, naive=True, notes=tablename)
        run(db, tablename, pp, klasses= [BDT], l=[0.5], c=cs, epsilon=[0.0001, 0.001], tau=[0.1, 0.55], p=0.7, notes=tablename)
        run(db, tablename, pp, klasses= [BDT], l=[0.5], c=[1.], epsilon=[0.01, 0.1], tau=[0.1, 0.55], p=0.7, notes=tablename)
        run(db, tablename, pp, klasses=[MR], l=[.5], c=cs, granularity=20, notes=tablename)

if __name__ == '__main__':
    from sqlalchemy import *
    db = create_engine('postgresql://localhost/dbwipes')
    init_db(db) 
    pp = PdfPages('figs/test.pdf')


    run_sigmod(pp, 2, 0.5)
    run_sigmod(pp, 3, 0.5)
    run_sigmod(pp, 4, 0.5)


    if True:
        pass
        # run and gather "ground truth" for everything
    #    run(db, 0, pp, klasses=[Naive], max_wait=10*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
    #    run(db, 11, pp, klasses=[Naive], max_wait=10*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
    #    run(db, 5, pp, klasses=[Naive], max_wait=10*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
    #    run(db, 15, pp, klasses=[Naive], max_wait=5*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)


        # run all others on intel_noon
    #    run(db, 0, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
    #    run(db, 0, pp, klasses= [BDT], l=[0.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.0001, 0.001], tau=[0.1, 0.45])
    #    run(db, 0, pp, klasses= [BDT], l=[0.5], c=[1.], epsilon=[0.01, 0.1], tau=[0.1, 0.45])
    #    run(db, 0, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=20)

        # run obama
    #    run(db, 11, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
    #    run(db, 11, pp, klasses=[BDT], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.0001, 0.001], tau=[0.1, 0.45])
    #    run(db, 11, pp, klasses=[BDT], l=[.5], c=[1.], epsilon=[0.01, 0.1], tau=[0.1, 0.45])
    #    run(db, 11, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=20)

        # run harddata 1
    #    run(db, 5, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
    #    run(db, 5, pp, klasses=[BDT], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.0001, 0.001], tau=[0.1, 0.5])
    #    run(db, 5, pp, klasses=[BDT], l=[.5], c=[1.], epsilon=[0.01, 0.1])
    #    run(db, 5, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=10)
    #    run(db, 5, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=20)

        # run harddata 15  -- high dim
    #    run(db, 15, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
    #    run(db, 15, pp, klasses=[BDT], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.0001, 0.001])
    #    run(db, 15, pp, klasses=[BDT], l=[.5], c=[1.], epsilon=[0.0001, 0.001, 0.01, 0.1])
    #    run(db, 15, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=10)
    #    run(db, 15, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=20)


    #    run(db, 0, pp, klasses=[NaiveMR], max_wait=20*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
    #    run(db, 11, pp, klasses=[NaiveMR], max_wait=20*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
    #    run(db, 5, pp, klasses=[NaiveMR], max_wait=20*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)
    #    run(db, 15, pp, klasses=[NaiveMR], max_wait=20*60, c=[0., .25, .5, .75, 1.], granularity=20, naive=True)



        # run naive on intel_noon and increase the available columns
        # will take forever...
    #    run(db, 0, pp, klasses=[Naive], max_wait=30*60, cols=['voltage'], granularity=20, naive=True)
    #    run(db, 0, pp, klasses=[Naive], max_wait=30*60, cols=['voltage', 'humidity'], granularity=20, naive=True)
    #    run(db, 0, pp, klasses=[Naive], max_wait=30*60, cols=['voltage', 'humidity', 'light'], granularity=20, naive=True)
    #    run(db, 0, pp, klasses=[Naive], max_wait=30*60, cols=['moteid'], granularity=20, naive=True)
    #    run(db, 0, pp, klasses=[Naive], max_wait=30*60, granularity=20, naive=True)


    pp.close()
