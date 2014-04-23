import pdb
from sqlalchemy import *
from monetdb import sql as msql
#import psycopg2
import sys

def connect(dbname, engine='pg'):
    try:
      if engine == 'monet' and dbname in ('intel', 'med'):
        db = msql.connect(user='monetdb', password='monetdb', hostname='localhost', database=dbname)
      else:
        conn = "postgresql://sirrice@localhost:5432/%s" % dbname
        db = create_engine(conn)
        #connection = "dbname='%s' user='sirrice' host='localhost' port='5432'" % (dbname)
        #db = psycopg2.connect(connection)
    except:
        sys.stderr.write( "couldn't connect\n")
        sys.exit()
    return db


def query(db, queryline, params=None):
    if db == None:
        return

    if 'monet' in str(db.__class__):
      res = db.cursor()
      if params:
        res.execute(queryline, params)
      else:
        res.execute(queryline)
    else:
      if params:
        res = db.execute(queryline, params)
      else:
        res = db.execute(queryline)
    try:
      for row in res.fetchall():
        yield row
    except:
      print queryline, params
      raise
    finally:
      res.close()

def close(db):
    if db != None:
        db.close()

