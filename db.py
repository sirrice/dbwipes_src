from sqlalchemy import *
#import psycopg2
import sys

def connect(dbname):
    try:
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

    if params:
      res = db.execute(queryline, params)
    else:
      res = db.execute(queryline)
    try:
      for row in res:
        yield row
    except:
      print queryline, params
      raise
    finally:
        res.close()

def close(db):
    if db != None:
        db.close()

