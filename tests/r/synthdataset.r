
library('ggplot2')

n = 1000
normal = data.frame(x=runif(n, 0, 100), y=runif(n, 0, 100), v=rnorm(n, 10, 5), g="Hold-out group")
outliers = data.frame(x=runif(n, 0, 100), y=runif(n, 0, 100), v=rnorm(n, 10, 5), g="Outlier group")
idxs = outliers$x >=30 & outliers$x<=70 & outliers$y >= 30 &outliers$y<=70
nn = sum(idxs)
outliers[idxs,] = data.frame(x=runif(nn, 30, 70), y = runif(nn, 30, 70), v = rnorm(nn, 90, 5), g="Outlier group")


data = rbind(normal, outliers)
data$A1 = data$x
data$A2 = data$y


pdf(file='synthdata.pdf', width=8, height=3)
qplot(A1, A2, data=data, color=v, geom='point', facets=.~g) + scale_color_continuous(lim=c(0, 100)) + opts(legend.title=theme_text(size=17), legend.text=theme_text(size=15), axis.text.x=theme_text(size=14), axis.text.y=theme_text(
size=14), strip.text.x=theme_text(size=15), strip.background=theme_blank(), axis.title.y=theme_text(size=17), axis.title.x=theme_text(size=17))
dev.off()
