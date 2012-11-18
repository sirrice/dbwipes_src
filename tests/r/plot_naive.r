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
clevels = c(0., 0.25, .5, .75, 1.)
dlevels = c('INTEL', 'EXPENSE', 'SYNTH-2D', 'HARD')
did2name = function(data) {
    data$datasetname = data$dataset
    data$datasetname[data$datasetname==0] = 'INTEL'
    data$datasetname[data$datasetname==5] = 'SYNTH-2D'
    data$datasetname[data$datasetname==11] = 'EXPENSE'
    data$datasetname[data$datasetname==15] = 'HARD'
return (data)
}

#
# Plot the naive curve for a given dataset:
# each line is a c value
#
query = "
select expid, klass, dataset, cols, c, cost, prec, epsilon, recall, f1, score, rule
from stats
where %s
"

where = "expid in (38, 39, 40)"

plot_naive = function(where){
    data = getData('dbwipes', sprintf(query, where))
    data$c = factor(data$c, clevels)
    data = did2name(data)
    data$datasetname = factor(data$datasetname, dlevels)

    plot = qplot(cost, f1, data=data, group=c, color=c, geom='line', facets=datasetname~.)
    plot = plot + opts(
      legend.position='bottom',
      legend.background = theme_rect(colour='white', fill='white', size=0)
    )
    return (plot)
}

printpdf = function(plot, fname) {
    pdf(file='naive.pdf', width=8, height=3)
    print(plot)
    dev.off()
}
