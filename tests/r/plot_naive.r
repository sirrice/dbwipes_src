library('ggplot2')
source('util.r')
library('scales')
library('stringr')
require(grid)

getData = function(dbname, sql) {
  drv = dbDriver('PostgreSQL')
  con = dbConnect(drv, dbname=dbname)
  res = dbGetQuery(con, sql)
  dbDisconnect(con)  
  res
}




dbname = 'dbwipes'
query = "
select expid, klass, dataset, cols, c, cost, prec, recall, f1, score, rule 
from stats
"

data = getData('dbwipes', query)


