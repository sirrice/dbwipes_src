library('ggplot2')
library('RPostgreSQL')
library(stringr)
require(grid)


theme_set(theme_bw())

getColumns = function(dbname, table) {
  drv = dbDriver('PostgreSQL')
  con = dbConnect(drv, dbname=dbname)

  q = "select column_name
       from information_schema.columns
       where table_name = '%s' and table_schema = 'public' and
       not(column_name in ('id', 'time', 'timestamp', 'uid', 'userid', 'rating'))"
  q = sprintf(q, table)
  res = dbGetQuery(con, q)
  dbDisconnect(con)

  res
}

getData = function(dbname, table, where) {
  drv = dbDriver('PostgreSQL')
  con = dbConnect(drv, dbname=dbname)
  
  if (where == '') {
    where = '1 = 1'
  }
  
  sql = "select *
         from %s
         where %s"
  q = sprintf(sql, table, where)
  res = dbGetQuery(con, q)
  dbDisconnect(con)  

  res
}


getGB = function(datas, attr, nbins=20) {
  print(c("get gb", attr))
  groups = datas[,attr]
  ugroups = unique(groups)

  if (is.numeric(ugroups) & length(ugroups) > nbins) {
    ret = getNumGB(groups, ugroups, nbins)
  } else {
    ret = getOrdGB(groups, ugroups, nbins)
  }

  ret
}

getOrdGB = function(groups, ugroups, nbins) {
  if (length(ugroups) <= nbins) {
    ret = function(vals) {
      vals
    }
    return(ret)
  }
  
  npergroup = ceiling((length(ugroups)/nbins))
  idxs = ceiling((1:length(ugroups))/npergroup)
  
  getidx = function(v) {
    idx = which(ugroups == v)[1]
    if (is.na(idx)) {
      idx = -1
    } else {
      idx = idxs[idx]
    }
    
    sprintf("group %d", idx)
  }

  ret = function(vals) {
    v = sapply(vals, getidx)
    v
  }
  ret
}

getNumGB = function(groups, ugroups, nbins) {
  if (max(ugroups) == min(ugroups)) {
    ret = function(v) {0}
    return(ret)
  }
  
  gbsd = sd(groups)
  gbm = mean(groups)
  gbmin = gbm - 2*gbsd
  gbmax = gbm + 2*gbsd
  gbmin = max(gbmin, min(ugroups[ugroups >= gbmin]))
  gbmax = min(gbmax, max(ugroups[ugroups < gbmax]))
  nbins = min(nbins, length(ugroups))
  gbbin = (gbmax - gbmin) / nbins
  bins = gbmin + (1:nbins)*gbbin

  ret = function(v) {
    idxs = sapply(v, function(s) max(1, min(nbins, (s-gbmin)/gbbin + 1)))
    bins[idxs]
  }

  ret
}

munge = function(data, aggattr, attr, agg) {
  groups = data$group
  ugroups = unique(groups)
  ngroups = length(ugroups)

  v0 = agg(data[, aggattr])
  err = function(foo)v0 - foo
  
  res = data.frame(gb=rep(NA, ngroups), v=rep(NA, ngroups), c=rep(NA, ngroups))

  for (i in 1:length(ugroups)) {
    group = ugroups[i]
    idxs = !(groups == group)
    vals = data[idxs, aggattr]

    v = err(agg(vals))
    n = length(groups) - sum(idxs)

    res[i,] = c(group, v, n)
  }

  res
}

plot = function(datas, aggattr, attr, agg, aggname, nbins=20) {
  res = get1DRes(datas, aggattr, attr, agg, aggname, nbins)
  if (is.na(res)) {
    return()
  }

  if (is.numeric(datas[,attr])) {
    plotAsLine(res, attr, aggname)
  } else {
    plotAsOrd(res, attr, aggname)
  }
  
}

get1DRes = function(datas, aggattr, attr, agg, aggname, nbins=20) {
  gbfunc = getGB(datas, attr, nbins)
  datas$group = gbfunc(datas[,attr])

  names = unique(datas$name)

  res = NA
  for (i in 1:length(names)) {
    name = names[i]
    data = datas[datas$name == name,]

    tmp = munge(data, aggattr, attr, agg)
    tmp$name = name

    
    if (i == 1) {
      res = tmp
    } else {
      res = rbind(res, tmp)
    }
  }

  if (sum(!is.nan(res$v)) == 0 || sum(!is.na(res$v)) == 0) {
    return(NA)
  }

  if (max(aggregate(rep(1, length(res$v)), list(res$name), sum)$x) == 1) {
    return(NA)
  }
  return(res)
  
}

plotAsLine = function(res, attr, aggname) {

  p = qplot(gb, v, data=res, group=name, color=name, geom=c('point','line'))
  p = p + scale_x_continuous(attr)
  p = p + opts(title=aggname)
  print(p)
}

plotAsOrd = function(res, attr, aggname) {
  res$v = as.numeric(res$v)
  res$gb = substr(res$gb, 1, 15)
  
  p = qplot(gb, v, data=res, geom='bar', position='dodge', color=name, fill=name) 
  p = p + opts(axis.text.x=theme_text(angle=-90, hjust=0))
  p = p + scale_x_discrete(attr)
  p = p + opts(title=aggname)
  print(p)
}



plot2D = function(datas, name, aggattr, attr1, attr2, agg, aggname, nbins=20) {
  gbf1 = getGB(datas, attr1, nbins)
  gbf2 = getGB(datas, attr2, nbins)
  datas$g1 = gbf1(datas[,attr1])
  datas$g2 = gbf2(datas[,attr2])

  data = datas[datas$name == name,]

  ug1 = unique(datas$g1)
  ug2 = unique(datas$g2)
  ngroups = length(ug1) * length(ug2)
  if (length(ug1) == 1 || length(ug2) == 1) {
    return(NA)
  }

  v0 = agg(data[,aggattr])
  err = function(foo)max(0, v0 - foo)

  res = data.frame(x=rep(NA, ngroups), y=rep(NA, ngroups), v=rep(NA, ngroups), c=rep(NA, ngroups))
  
  for (i in 1:length(ug1)) {
    for (j in 1:length(ug2)) {
      
      idxs = !(datas$g1 == ug1[i] & datas$g2 == ug2[j])
      vals = data[idxs, aggattr]
      vals = vals[!is.na(vals)]

      v = err(agg(vals))
      n = length(vals)
      if (n == 0) {
        v = NA
      } else {
        if(is.na(v)) {
          browser()
        }
      }

      
      res[(i-1) * length(ug2) + j,] = c(ug1[i], ug2[j], v, n)

    }
  }

  if (sum(!(is.na(res$v)) == 0)) {
    return(NA)
  }
  res = res[!is.na(res$v),]
  xsize = (max(ug1) - min(ug1)) / (1+length(ug1))
  ysize = (max(ug2) - min(ug2)) / (1+length(ug2))
  res$xsize = xsize
  res$ysize = ysize

  print(c("plotting", length(ug1), length(ug2)))
  #p = ggplot(res, aes(xmin = x, xmax = x + xsize, ymin = y, ymax = y + ysize, color=v, fill=v)) + geom_rect()
  res$y = paste(res$y)
  p = qplot(x, v, data=res, group=y, geom='jitter', fill=y, size=3, color=y)
  p = p + scale_colour_hue()
  #p = p + scale_fill_gradient(low="red", high="green")  
  p = p + scale_x_continuous(attr1)
  p = p + scale_y_continuous('delta')
  p = p + opts(title=sprintf("%s\t%s",aggname, attr2))
  
  print(p)
  
}


