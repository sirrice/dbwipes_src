#from gevent.pywsgi import WSGIServer # must be pywsgi to support websocket
#from geventwebsocket.handler import WebSocketHandler
from flask import Flask, request, render_template, g, redirect
import json
import md5
import traceback
from datetime import datetime
from monetdb import sql as msql

from db import *
from arch import *
from util import *
from parallel import *
import pdb
app = Flask(__name__)




@app.before_request
def before_request():
  dbname = request.form.get('db', 'intel')
  g.dbname = dbname
  g.tstamp = md5.md5(str(hash(datetime.now()))).hexdigest()
  g.db = connect(dbname, engine='pg')
  try:
    #g.monetdb = msql.connect(user='monetdb', password='monetdb', hostname='localhost', database=dbname)
    g.monetdb = connect(dbname, engine='monet')
  except:
    g.monetdb = None
      

@app.teardown_request
def teardown_request(exception):
  try:
    g.db.close()
  except:
      pass

  try:
    g.monetdb.close()
  except:
    pass


def run_sql_query(parsed, orig_where):
  ret = []
  sql = parsed.to_outer_join_sql(orig_where)
  if g.monetdb:
    try:
      res = g.db.execute("%s limit 0" % str(parsed))
      keys = res.keys()
      res = g.monetdb.cursor()
      res.execute(sql)
      #keys = zip(*res.description)[0]
    except:
      res = g.db.execute(sql)
      keys = res.keys()
  else:
    res = g.db.execute(sql)
    keys = res.keys()

  try:
    for row in res.fetchall():
      ret.append(dict(zip(keys, row)))
  finally:
    res.close()
  dt_labels = find_datetime_labels(ret)
  return ret


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





        
@app.route('/debug/', methods=["GET"])
@app.route('/', methods=["POST", "GET"])
def index():
    context = dict(request.form.items())
    try:
        sql = request.form.get('query', '')
        delids = json.loads(request.form.get('bad_tuple_ids', '{}')) # ids to delete
        where = request.form.get('filter', '') 

        if not sql:
            sql = """ SELECT avg(temp), stddev_samp(temp), hr
            FROM readings
            GROUP BY hr; """
            #WHERE date > '2004-3-1' and date < '2004-3-15'

        obj = get_query_sharedobj(sql, delids)
        if where.strip():
            obj.parsed.where.append(where)
        print str(obj.parsed)
        data_and_labels = run_query(obj)

        context['query'] = obj.parsed.prettify()
        context['data'] = json.dumps(data_and_labels['data'], default=json_handler)
        context['labels'] = json.dumps(data_and_labels['labels'])
        context['result_schema'] = [ (attr, t.__name__, True) for attr, t in obj.schema.items()]
        if 'data' in context:
          print "data has %d points" % (len(context['data']))
    except Exception as e:
        import traceback
        traceback.print_exc()
        context['errormsg'] = str(e)
    return render_template('index.html', **context)


@app.route('/json/facets/', methods=['POST', 'GET'])
def json_facets():
  data = {}

  try:
    db = request.form.get('db', None)
    query = request.form.get('query', None)

    try:
      print query
      parsed = parse_sql(g.db, query)
      where = ' and '.join(parsed.where)
      table = parsed.fr[0]
    except:
      where = ''
      table = ''

    from summary import Summary
    s = Summary(g.db, table)
    cols = s.get_columns()

    # filter for non-agg and non-gb cols

    typs = map(s.get_type, cols)
    cardinalities = map(s.get_cardinality, cols)
    groups = [s.get_col_groupby(*e) for e in zip(cols, typs)]
    col_datas = []
    for c,t,ca,gb in zip(cols, typs, cardinalities, groups):
      print "col: %s" % c
      if gb:
        stats = s.get_group_stats(c,gb)
        stats = [{'val': val, 'count': count} for (val,count) in stats]
        el = {
          'col': c,
          'typ': t,
          'card': ca,
          'gb': gb,
          'hist': stats
        }
        col_datas.append(el)

    data = {
      'col_datas': col_datas
    }

  except:
    traceback.print_exc()

  return json.dumps(data, default=json_handler)

    
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
  ret = {'data' : {}, 'query':None};
  try:
    sql = request.form['query']
    where = request.form.get('filter', None)
    parsed = parse_sql(g.db, sql)
    orig_where = list(parsed.where)
    if where:
      parsed.where.append(where)
    print str(parsed)[:200]
    data = run_sql_query(parsed, orig_where)
    ret['data'] = data
    ret['query'] = str(parsed)
  except Exception as e:
    import traceback
    traceback.print_exc()
  return json.dumps(ret, default=json_handler)

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

    

@app.route('/debug/', methods=["POST"])
def debug():
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



        start = time.time()
        parallel_debug(
          obj,
          nprocesses=min(maxkeys, 4),
          parallelize=True,
          nstds=0,
          errperc=0.001,
          epsilon=0.008,
          msethreshold=0.15,
          c=obj.c,
          complexity_multiplier=4.5,
          l=0.9,
          max_wait=10,
          DEBUG=True
        )
        cost = time.time() - start
        print "end to end took %.4f" % cost



        context.update( {
          'query': obj.prettify_sql,
          'errtype' : str(obj.errors[0].errtype),
          'erreq' : 'erreq',
          'data' : json.dumps(data, default=json_handler),
          'labels' : json.dumps(labels)
        })
        context['result_schema'] = [ 
            (attr, t.__name__, attr not in obj.ignore_attrs)
            for attr, t in obj.schema.items() 
        ]

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
  nrules = 6 
  for label, clauses in obj.clauses.items():
    rules = [p[0] for p in obj.rules[label]]
    clauses = [c.strip() for c in clauses]
    #clauses = filter(lambda e:e.strip(), rm_dups(clauses, hash))
    for rule, clause in zip(rules[:nrules], clauses[:nrules]):
      # print "\t", clause

      tmpq = obj.parsed.clone()
      if clause:
        cwhere = 'not (%s)' % clause 
        tmpq.where.append(cwhere)
      else:
        cwhere = ''
      clause_parts = rule.toCondStrs()
      print rule
      print clause_parts

      equiv_clause_parts = [r.toCondStrs() for r in rule.cluster_rules]

      filter_opts[label].append( (clause_parts, str(tmpq), cwhere, json.dumps({}), equiv_clause_parts, '%.4f' % rule.quality, [round(rule.c_range[0], 3), round(rule.c_range[1], 3)], idx) )
      idx += 1
  return filter_opts



if __name__ == "__main__":
    DEC2FLOAT = psycopg2.extensions.new_type(
        psycopg2.extensions.DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None)
    psycopg2.extensions.register_type(DEC2FLOAT)
    app.debug = True
    app.run(port=8000)

    #address = ('', 8000)
    #http_server = WSGIServer(address, app)#, handler_class=WebSocketHandler)
    #print "running"
    #http_server.serve_forever()
