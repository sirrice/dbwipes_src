#
# For sigmod SYNTH Tests
# 1) compute synth for 2/3/4-d with medium and high outlier boxes
# 2) compute the two c values and store it
# 3) compute and store the boxes
# 4) execute suite of tests for each dataset
#
import pdb
import sys
import math
import json
import random
sys.path.extend(['.', '..'])

import numpy as np

from sqlalchemy import *
from common import *
from test import init_db
from util import get_logger
from misc.gensinglecluster import gen_points, in_box


random.seed(0)

format_list = lambda fmt, l: [fmt % (isinstance(p, list) and tuple(p) or p) for p in l]


def generate_datasets(db, ndim, uo=30):
    random.seed(0)
    """
    generates the 2/3/4-d datasets and stores them in db: sigmod
    """
    mid_box, high_box, schema, generator = gen_points(2000, ndim, ndim, 0.25, 10, 10, uo, 10)
    pts = [pt for pt in generator]
    c_mid, c_high = compute_c_values(pts, mid_box, high_box)

    # store boundaries
    tablename = 'data_%d_%d' % (ndim, int(uo))
    id = add_config(db, tablename, 2000, ndim, ndim, 0.25, 10, 10, uo, 10, mid_box, high_box, c_high)

    # save the damn thing
    save_pts(db, tablename, pts)
    

def setup(db):
    try:
        with db.begin() as conn:
            conn.execute("""
            create table synth (
                id serial,
                tablename varchar(128),
                npts int,
                ndim int,
                kdim int,
                volperc float,
                uh float,
                sh float,
                uo float,
                so float,
                mid text,  -- json encoded boundary for mid level
                high text,  -- json encoded
                c float -- threshold for picking  high over mid
            )
            """)
    except:
        import traceback
        traceback.print_exc()
        pass

def save_pts(db, tablename, pts):
    ndims = len(pts[0]) - 2
    with db.begin() as conn:
        q = """
        create table %s (
        id serial,
        %s,
        g float,
        v float)""" % (tablename, ','.join(format_list('a%d float', xrange(ndims))))
        conn.execute(q)

        pts_str = []
        for pt in pts:
            s = '(%s)' % ','.join(['%f']*(ndims+2)) % tuple(pt)
            pts_str.append(s)


        bigq = """insert into %s (%s) values
           %s""" % (tablename, ','.join(format_list('a%d', xrange(ndims))+['g','v']), ','.join(pts_str))
        conn.execute(bigq)


def add_config(db, *args):
    args = list(args)
    with db.begin() as conn:
        q = """
        insert into synth (tablename, npts, ndim, kdim, volperc, uh, sh, uo, so, mid, high, c)
        values (%s) returning id""" % ((',%s' * 12)[1:])
        args[-3] = json.dumps(args[-3])
        args[-2] = json.dumps(args[-2])
        id = conn.execute(q, *args ).fetchone()[0]
        return id

def all_configs(db):
    with db.begin() as conn:
        q = """ select * from synth"""
        for config in conn.execute(q).fetchall():
            config = list(config)
            config[-3] = json.loads(config[-3])
            config[-2] = json.loads(config[-2])
            yield config



def get_config(db, tablename):
    with db.begin() as conn:
        q = """ select * from synth where tablename = %s"""
        config = conn.execute(q, tablename).fetchone()
        config = list(config)
        config[-3] = json.loads(config[-3])
        config[-2] = json.loads(config[-2])
        return config

def get_pts(db, tablename):
    with db.begin() as conn:
        q = """ select * from %s""" % tablename
        return [pt[1:] for pt in conn.execute(q).fetchall()]

def get_ids_in_bounds(db ,tablename, bounds):
    with db.begin() as conn:
        where = ['%f <= a%d and a%d <= %f' % (minv, i, i, maxv) for i, (minv, maxv) in enumerate(map(tuple, bounds))]
        where = ' and '.join(where)
        q = """ select id from %s where %s""" % (tablename, where)
        return [int(row[0]) for row in conn.execute(q)]




def compute_c_values(pts, mid_bounds, high_bounds, f=np.mean):
    orig_f = f
    f = lambda pts: orig_f([pt[-1] for pt in pts])
    pts = set(map(tuple,pts))
    all_vs, mid_vs, high_vs = set(), set(), set()
    n_mid, n_high = 0, 0
    for pt in pts:
        if pt[-2] != 8:
            continue
        all_vs.add(pt)
        if in_box(pt[:-2], mid_bounds):
            mid_vs.add(pt)
        if in_box(pt[:-2], high_bounds):
            high_vs.add(pt)


    orig = f(all_vs)
    nomid = f(all_vs.difference(mid_vs))
    nohigh = f(all_vs.difference(high_vs))

    dm = orig - nomid
    dh = orig - nohigh

    cm = len(mid_vs)
    ch = len(high_vs)
    
    highc = math.log(dh / dm) / math.log(1.*ch/cm)
    
    highcs = []
    for i in [1, 50, 100]:# xrange(1, 100, 10):
        for n in xrange(100):
            mid_samp = random.sample(mid_vs.difference(high_vs), i)#random.randint(1, 100))
            high_samp = random.sample(high_vs,i)# random.randint(1, 100))
            mix_samp = mid_samp + high_samp
            dm = orig - f(all_vs.difference(mix_samp)) 
            dh = orig - f(all_vs.difference(high_samp))
            ch, cm = float(len(high_samp)), float(len(mix_samp))
            if dh <= 0 or dm <= 0 or dh == dm or ch == cm:
                continue
            highc = math.log(dh / dm) / math.log(ch/cm)
            highcs.append(highc)
#            print '%d\t%.4f\t%.4f\t%.4f' % (i, highc, dh / 1., dm / 2.**highc)
    highc = np.mean(highcs)
    print '%d\t%.4f\t%.4f\t%.4f\t%.4f' % (i, highc, np.std(highcs), min(highcs), max(highcs))
    return highc - .25*np.std(highcs), highc + .25*np.std(highcs)

    return highc

def save_result(db, total_cost, costs, stats, rule, ids, dataset, notes, kwargs):
    acc, p, r, f1 = stats
    ids_str = ','.join(map(str, ids))
    isbest = rule.isbest
    vals = [0, dataset, notes]
    pkeys = ['klassname', 'cols', 'epsilon', 'c', 'lambda']
    vals.extend([kwargs.get(key, None) for key in pkeys])
    vals.extend((total_cost, acc, p, r, f1, rule.quality, isbest, str(rule), ids_str))
    stat = vals


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




def run(sigmoddb, statsdb, tablename, bounds, **kwargs):
    params = {
            'epsilon' : 0.0005,
            'tau' : [0.1, 0.5],
            'p' : 0.7,
            'l' : 0.5,
            'min_improvement' : 0.01,
            'c' : 0.0,
            'max_wait' : 20 * 60,
            'klass' : BDT
            }

    params.update(kwargs)
    klassname = params['klass'].__name__
    params['klassname'] = klassname
    params['dataset'] = tablename


    costs, rules, all_ids, table_size, learner = run_experiment(tablename, **params)
    truth = set(get_ids_in_bounds(sigmoddb, tablename, bounds))
    all_stats = [compute_stats(ids, truth,  table_size) for ids in all_ids]

    if klassname == 'Naive':
        rules = rules[:1]
        all_ids = all_ids[:1]

    for stats, rule, ids in zip(all_stats, rules, all_ids):
        save_result(statsdb, costs['cost_total'], costs, stats, rule, ids, tablename, tablename, params)

    if klassname == 'Naive':
        for timein, rule in learner.checkpoints:
            chk_ids = get_row_ids(rule, learner.full_table)
            stats = compute_stats(chk_ids, truth, table_size)
            save_result(statsdb, timein, {}, stats, rule, chk_ids, tablename, tablename, params)


def run_tests(sigmoddb, statsdb, **params):
    """
    defaults: 
     epsilon: 0.001
     tau: [0.1, 0.5]
     p: 0.7
     lambda: 0.5
     min_improvement: 0.01
    """

    for ndim in  [2,3,4]:
        for uo in [30, 80]:

            tablename = "data_%d_%d" % (ndim, uo)
            config = get_config(sigmoddb, tablename)
            mid_bounds = config[-3]
            high_bounds = config[-2]
            pts = get_pts(sigmoddb, tablename)
            c_mid, c_high = compute_c_values(pts, config[-3], config[-2])

            for klass in [Naive, BDT, NDT, MR]:
                for c in [c_mid, c_high]:
                    run(sigmoddb, statsdb, tablename, high_bounds, c=c, klass=klass, **params)

statsdb = create_engine('postgresql://localhost/dbwipes')
sigmoddb = create_engine('postgresql://localhost/sigmod')
init_db(statsdb)
tablename = 'data_2_30'
config = get_config(sigmoddb, tablename)
mid_bounds = config[-3]
high_bounds = config[-2]
pts = get_pts(sigmoddb, tablename)
c_mid, c_high = compute_c_values(pts, config[-3], config[-2])
print c_mid, c_high




run_tests(sigmoddb, statsdb, max_wait=30*60)

if False:
    setup(db)
    for uo in [30, 40, 50, 60, 70, 80, 90, 100]:
        generate_datasets(db, 2, uo=uo)
        generate_datasets(db, 3, uo=uo)
        generate_datasets(db, 4, uo=uo)
