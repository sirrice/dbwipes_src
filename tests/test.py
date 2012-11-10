import sys
import pdb
sys.path.extend(('.', '..'))


from common import *



    # configurations
    # ndt: nothing, c
    # basic: max complexity, c
    # bdt: epsilon, tau, lambda, c
    # mc: c
def print_stats(pp, stats, title):
    fig = plt.figure(figsize=(10, 3))
    accs, ps, rs, ss, rule = zip(*stats)
    for idx, (ys, statname) in enumerate(((accs, 'acc'), (ps, 'prec'), (rs, 'recall'))):
        sub = fig.add_subplot(int('13'+str(idx)))
        sub.scatter(ss, ys, lw=0)

        sub.set_ylim(0, 1.)
        sub.set_title('%s:%s' % (title, statname))

    plt.savefig(pp, format='pdf')


def make_params(**kwargs):
    ret = []
    for klass in kwargs.get('klasses', [NDT, BDT, MR, Naive]):
        for c in kwargs.get('c', [0, 0.5, 0.75, 1.]):

            params = {'c':c, 'klass':klass, 'klassname' : klass.__name__[-7:]}
            params['cols'] = kwargs.get('cols', None)

            if klass == BDT:
                for epsilon in kwargs.get('epsilon', [0.001, 0.01, 0.05]):
                    for l in kwargs.get('l', [0.5, 1.]):
                        params2 = dict(params)
                        params2.update({ 
                            'epsilon':epsilon, 
                            'lambda':l, 
                            'min_improvement' : .01})
                        ret.append(params2)
            elif klass == Naive:
                params2 = dict(params)
                params2['max_complexity'] = 2
                params2['granularity'] = 5
                ret.append(params2)
            else:
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


def run(expid, datasetidx, pp=None, **kwargs):
    pkeys = ['klassname', 'cols', 'epsilon', 'c', 'lambda']
    for params in make_params(**kwargs):
        cost, costs, bstats, mstats, astats = run_experiment(datasetidx, **params)
        thresh = min(bstats, key=lambda s: s[-2])[-2]
        for acc, p, r, s, rule in mstats:
            f1 = (p+r) > 0 and 2 * (p*r) / (p + r) or 0.

            vals = [expid, datasetidx] 
            vals.extend([params.get(key, None) for key in pkeys]) 
            vals.extend((cost, acc,  p, r, f1, s, bool(s >= thresh), str(rule)))
            save_result(db, vals, costs)
            print mkfmt(vals) % tuple(vals)

        if pp:
            print_stats(pp, mstats, ','.join(map(lambda p: '%s:%s'%tuple(p), params.items())))


def save_result(db, stat, costs):
    with db.begin() as conn:
        q = """insert into stats(expid, dataset, klass, cols, epsilon, c,     
                                 lambda, cost, acc, prec, recall, f1, score, 
                                 isbest, rule) values(%s) returning id""" % (','.join(['%s']*len(stat)))
        sid = conn.execute(q, *stat).fetchone()[0]

        q = """insert into costs(sid, name, cost) values(%s,%s,%s)"""
        for name, cost in costs.items():
            conn.execute(q, sid, name, cost)


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
        dataset int,
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
        completed bool,
        rule text null
                )""")
            conn.execute("""create table costs (
            id serial,
            sid int,
            name varchar(128),
            cost float)""")
    except:
        pass

from sqlalchemy import *
db = create_engine('postgresql://localhost/dbwipes')
init_db(db) 
pp = PdfPages('figs/test.pdf')
expid = nextexpid(db)
print "experiment:", expid


run(expid, 5, pp, klasses=[BDT], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.001, 0.0001], tau=[0.09, 0.3])

if True:
    # run all others on intel_noon
    run(expid, 0, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
    run(expid, 0, pp, klasses= [BDT], l=[0.5], c=[1.], epsilon=[0.0001, 0.001, 0.01, 0.1])
    run(expid, 0, pp, klasses= [BDT], l=[0.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.001])
    run(expid, 0, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.])
    run(expid, 0, pp, klasses=[MR], max_wait=30*60, c=[0.], granularity=[20], naive=True)
    complete(db, expid)
    expid += 1

    # run obama
    run(expid, 11, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
    run(expid, 11, pp, klasses=[BDT], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.0001, 0.001])
    run(expid, 11, pp, klasses=[BDT], l=[.5], c=[1.], epsilon=[0.0001, 0.001, 0.01, 0.1])
    run(expid, 11, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.])
    run(expid, 11, pp, klasses=[MR], max_wait=30*60, c=[0.], granularity=[20], naive=True)
    complete(db, expid)
    expid += 1

    # run harddata 1
    run(expid, 5, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
    run(expid, 5, pp, klasses=[BDT], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.0001, 0.001])
    run(expid, 5, pp, klasses=[BDT], l=[.5], c=[1.], epsilon=[0.0001, 0.001, 0.01, 0.1])
    run(expid, 5, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=[20])
    run(expid, 5, pp, klasses=[MR], max_wait=30*60, c=[0.], granularity=[20], naive=True)
    complete(db, expid)
    expid += 1

    # run harddata 15  -- high dim
    run(expid, 15, pp, klasses=[NDT], c=[0., 0.25, 0.5, 0.75, 1.])
    run(expid, 15, pp, klasses=[BDT], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], epsilon=[0.0001, 0.001])
    run(expid, 15, pp, klasses=[BDT], l=[.5], c=[1.], epsilon=[0.0001, 0.001, 0.01, 0.1])
    run(expid, 15, pp, klasses=[MR], l=[.5], c=[0., 0.25, 0.5, 0.75, 1.], granularity=[20])
    run(expid, 15, pp, klasses=[MR], max_wait=30*60, c=[0.], granularity=[20], naive=True)
    complete(db, expid)
    expid += 1



    # run naive on intel_noon and increase the available columns
    # will take forever...
    run(expid, 0, pp, klasses=[MR], max_wait=30*60, cols=['voltage'], granularity=[20], naive=True)
    run(expid, 0, pp, klasses=[MR], max_wait=30*60, cols=['voltage', 'humidity'], granularity=[20], naive=True)
    run(expid, 0, pp, klasses=[MR], max_wait=30*60, cols=['voltage', 'humidity', 'light'], granularity=[20], naive=True)
    run(expid, 0, pp, klasses=[MR], max_wait=30*60, cols=['moteid'], granularity=[20], naive=True)
    run(expid, 0, pp, klasses=[MR], max_wait=30*60, granularity=[20], naive=True)
    complete(db, expid)
    expid += 1


complete(db, expid)
pp.close()
