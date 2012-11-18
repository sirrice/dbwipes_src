
# get plot of a given experiment


# extract dimension and u_o info from notes for naive
select distinct expid, 
                substring(dataset from 6 for 1)::int as ndim, 
                substring(dataset from 8 for 1)::int as kdim, 
                substring(dataset from 20 for 2)::int as uo, c, char_length(ids) from stats where klass = 'Naive' and notes is not null order by ndim, kdim, c;


# plot naive for 2x2, 3x3, 4x4
qplot(cost, f1, data=plot_naive('expid in (85, 90, 115, 120, 155)')$data, color=c, group=c, geom='line', facets=dataset~.)
> pdf(file='naive.pdf', width=8, height=40); print(qplot(cost, f1, data=plot_naive('klass = \'Naive\'')$data, color=c, group=c, geom='line', facets=dataset~.) + scale_x_continuous(lim=c(0, 350))); dev.off()


# plot BDT, NDT, MR for dataset 15
qplot(score, f1, data=plot_naive('expid in (51,60,26, 59, 55)')$data, group=epsilon, color=epsilon, geom='point', facets=klass~c)


# plot BDT, NDT, MR for dataset 11
pdf(file='expenses.pdf', width=8, height=40); print(qplot(prec, recall, data=plot_naive('expid in (53, 48, 57, 47,
                24)')$data, group=epsilon, color=epsilon, geom='point', facets=klass~c)); print(qplot(score, f1, data=plot_naive('expid in (53, 48, 57, 47, 24)')$data, group=epsilon, color=epsilon, geom='point', facets=klass~c)); dev.off()


