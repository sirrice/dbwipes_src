from gevent.pywsgi import WSGIServer # must be pywsgi to support websocket
from geventwebsocket.handler import WebSocketHandler
from flask import Flask, request, render_template, g, redirect
import json
import md5
import traceback
from datetime import datetime
from db import *
from arch import *
from util import *
from parallel import *
import pdb
app = Flask(__name__)




@app.before_request
def before_request():
    dbname = request.form.get('db', 'intel')
    g.db = connect(dbname)
    g.dbname = dbname
    g.tstamp = md5.md5(str(hash(datetime.now()))).hexdigest()
        

@app.teardown_request
def teardown_request(exception):
    try:
        g.db.close()
    except:
        pass


def run_query(obj):
    res = list(obj.get_agg_dicts())
    toshort = lambda attr : attr.shortname
    axes_labels = map(toshort, obj.parsed.select.nonaggs)
    plot_labels = map(toshort, obj.parsed.select.aggs)
    dt_labels = find_datetime_labels(res)
    id_label = 'id'
    labels = {'gbs' : axes_labels,
              'x' : axes_labels[0],
              'aggs' : plot_labels,
              'id' :   id_label,
              'dt' : dt_labels}

    return {'data' : res, 'labels' : labels}

def find_datetime_labels(data):
    dt_labels = set()
    for pt in data[:500]:
        for key, val in pt.iteritems():
            if hasattr(val, 'isoformat'):
                dt_labels.add(key)
    return list(dt_labels)


def replace_keys(dictlist, replace, stringify=True):
    if stringify:
        f = lambda d: dict([(str(replace.get(k,k)), v) for k, v in d.iteritems()])
    else:
        f = lambda d: dict([(replace.get(k,k), v) for k, v in d.iteritems()])
    return map(f, dictlist)

def construct_id_ranges(ids):
    """
    given a bag of (possibly string) ids, return minimum list of ranges that describe
    the ids
    assumes an id is fundamentally an integer
    """
    ids = list(map(int, ids))
    ids.sort()
    ranges = []
    for tid in ids:
        if not len(ranges) or ranges[-1][-1] + 1 != tid:
            ranges.append([tid, tid])
        ranges[-1][-1] = tid
    return ranges

def get_query_sharedobj(sql, delids):
    if delids:
        allids = set()
        map(allids.update, delids.values())
        ranges = construct_id_ranges(allids)
        where = map(lambda r: '(%d <= id and id <= %d)' %(r[0], r[1]), ranges)
        where = 'not (%s)' % ' or '.join(where)
        print where

    obj = SharedObj(g.db, sql)
    if delids:
        obj.parsed.where.append(where)
    return obj

def json_handler(obj):
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    raise TypeError, 'Object of type %s with value of %s is not JSON serializable' % (type(obj), repr(obj))





        
@app.route('/', methods=["POST", "GET"])
def intel_query():
    context = dict(request.form.items())
    try:
        sql = request.form.get('query', '')
        delids = json.loads(request.form.get('bad_tuple_ids', '{}')) # ids to delete
        print delids
        where = request.form.get('filter', '') 

        if not sql:
            sql = """SELECT avg(temp), stddev(temp), date_trunc('hour',(date) + (time)) as dt
            FROM readings
            WHERE (((date) + (time)) > '2004-3-1') and (((date) + (time)) < '2004-3-5')
            GROUP BY dt
            ORDER BY dt ASC"""

        obj = get_query_sharedobj(sql, delids)
        if where.strip():
            obj.parsed.where.append(where)
        data_and_labels = run_query(obj)

        context['query'] = obj.parsed.prettify()
        context['data'] = json.dumps(data_and_labels['data'], default=json_handler)
        context['labels'] = json.dumps(data_and_labels['labels'])
        context['result_schema'] = [ (attr, t.__name__, True) for attr, t in obj.schema.items()]
    except Exception as e:
        import traceback
        traceback.print_exc()
        context['errormsg'] = str(e)
    return render_template('index.html', **context)

    
@app.route('/json/schema/', methods=["POST", "GET"])
def json_schemas():
    data = defaultdict(list)
    try:
        tquery = "select tablename from pg_tables where schemaname = 'public'"
        squery = '''select column_name, data_type
        from information_schema.columns
        where table_name = %s;'''
        
        for table, in query(g.db, tquery):
            data[table] = map(tuple, query(g.db, squery, (table,)))
    except:
        traceback.print_exc()
    return json.dumps(data, default=json_handler)
    


@app.route('/json/query/', methods=["POST", "GET"])
def json_query():
    data_and_labels = {'data' : {}, 'labels' : {}}
    try:
        sql = request.form['query']
        delids = json.loads(request.form.get('bad_tuple_ids', '{}')) # ids to delete
        where = request.form.get('filter', '')
        obj = get_query_sharedobj(sql, delids)
        if where.strip():
            obj.parsed.where.append(where)
        print "/json/query.Final", str(obj.parsed)
        data_and_labels = run_query(obj)

        print data_and_labels['data'][:5]
        
        
    except Exception as e:
        import traceback
        traceback.print_exc()
    return json.dumps(data_and_labels, default=json_handler)


@app.route('/json/filterq/', methods=["POST", "GET"])
def json_filterq():
    data_and_labels = {'data' : {}, 'labels' : {}}
    try:
        # extract and pool together keys from data
        data = json.loads(request.form['data'])
        sql = request.form['query']        
        where = request.form.get('filter', '')
        obj = SharedObj(g.db, sql)

        
        keys = set()
        map(lambda v: keys.update(v), data.values())
        keys = list(keys)
        if where.strip():
            obj.parsed.where.append(where)

        data_and_labels = run_base_tuple_query(obj, keys)
        
    except:
        import traceback
        traceback.print_exc()
    return json.dumps(data_and_labels, default=json_handler)


def run_base_tuple_query(obj, keys):
    # we want
    # 1) expression in each aggregate function
    # 2) group by value
    toshort = lambda attr: attr.shortname
    toexpr = lambda agg: str(agg.expr)
    gb_labels = map( toshort, obj.parsed.select.nonaggs)
    agg_labels = set()
    map(lambda agg: agg_labels.update(agg.cols), obj.parsed.select.aggregates)
    #agg_labels = map( toexpr, obj.parsed.select.aggregates )
    agg_labels = list(agg_labels)
    id_label = 'id'
    sel = agg_labels + obj.parsed.select.nonaggs + ['id']

    data = list(obj.get_filter_dicts(keys=keys, attrs=sel))
    labels = {'gbs' : gb_labels,
              'x' : gb_labels[0],
              'aggs' : agg_labels,
              'id' : 'id'}
    return {'data' : data, 'labels' : labels}

    

@app.route('/debug/', methods=["POST", "GET"])
def intel_debug():
    context = dict(request.form.items())
    try:    
        if request.method == 'POST':
            obj = parse_debug_args(g.db, request.form, dbname=g.dbname)
            qobj = obj.parsed
            data_and_labels = run_query(obj)
            data, labels = data_and_labels['data'], data_and_labels['labels']

            maxkeys = 0
            for aggerr in obj.errors:
                print aggerr.agg.shortname, '\t', aggerr.keys
                maxkeys = max(maxkeys, len(aggerr.keys))



            parallel_debug(obj,
                           nprocesses=min(maxkeys, 4),
                           parallelize=True,
                           nstds=0,
                           errperc=0.001,
                           epsilon=0.05,
                           msethreshold=0.15,
                           complexity_multiplier=4.5)



            context.update( {'query': obj.prettify_sql,
                             'errtype' : str(obj.errors[0].errtype),
                             'erreq' : 'erreq',
                             'data' : json.dumps(data, default=json_handler),
                             'labels' : json.dumps(labels)})
            context['result_schema'] = [ (attr, t.__name__, attr not in obj.ignore_attrs)
                                         for attr, t in obj.schema.items() ]


            # recompute aggregate query with classifier suggested predicates
            filter_opts = create_filter_options(obj)
            context['filter_opts'] = filter_opts.items()
    except Exception as e:
        import traceback
        traceback.print_exc()
        context['errormsg'] = str(e)
    return render_template('index.html', **context)

def create_filter_options(obj):

    filter_opts = defaultdict(list)
    idx = 0
    for label, clauses in obj.clauses.items():
        rules = [p[0] for p in obj.rules[label]]
        clauses = filter(lambda e:e.strip(), rm_dups(clauses, hash))
        for rule, clause in zip(rules[:6], clauses[:6]):
            # print "\t", clause

            tmpq = obj.clone()
            cwhere = 'not (%s)' % clause 
            tmpq.add_where( cwhere )
            clause_parts = [c.strip() for c in str(rule).split(' and ')]

            filter_opts[label].append( (clause_parts, tmpq.sql, cwhere, json.dumps({}), idx) )
            idx += 1
    return filter_opts



if __name__ == "__main__":
    DEC2FLOAT = psycopg2.extensions.new_type(
        psycopg2.extensions.DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None)
    psycopg2.extensions.register_type(DEC2FLOAT)
    app.debug = True
    address = ('', 8000)
    http_server = WSGIServer(address, app, handler_class=WebSocketHandler)
    print "running"
    http_server.serve_forever()
