

if __name__ == '__main__':
    import sys
    sys.path.extend( ['.', '..'])
    from util import *
    
    import random
    import timeit
    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('figs/linearfit.pdf')
    import matplotlib.pyplot as plt

    def f(x, a, b):
        return a + b*x
    def err(a,b,xs,ys):
        e = (ys - (a+b*xs))
        return (np.dot(e,e) / (len(xs) - 2.)) ** 0.5
    

    pts = []
    for i in xrange(1000):
        x,y = i/10., (i/10. + random.random()*2)
        pts.append( (x,y) )
    for i in xrange(30):
        pts.append( (100, 5*random.random()) )
    pts = np.array(pts)

    fig = plt.figure()
    sub = fig.add_subplot(111)

    
    sub.scatter(pts[:,0], pts[:,1], s=2, lw=0, alpha=0.4)
    xs, ys = pts[:,0], pts[:,1]

    a,b = curve_fit(f, pts[:, 0], pts[:, 1])[0]
    t = timeit.Timer(lambda: curve_fit(f, pts[:, 0], pts[:, 1]))
    t.timeit(10)
    print t.timeit(1000) / 1000.
    print "err", err(a,b,xs,ys)

    lf = LinearFit()
    a,b,r2 = lf(pts)
    sub.plot(xs, a+xs*b, c='blue')
    t = timeit.Timer(lambda: lf(pts))
    print t.timeit(1000) / 1000.
    sub.plot(xs, a+xs*b, c='green')    
    print "err", err(a,b,xs,ys)

    t = timeit.Timer(lambda: (lf.add(pts), lf.rm(pts) ) )
    print t.timeit(1000) / 1000.
    a,b,r2 = lf.update_fit()
    sub.plot(xs, a+xs*b, c='red')
    print "err", err(a,b,xs,ys)    

    # t = timeit.Timer(lambda: lf.rm(pts))
    # t.timeit(1000) / 1000.
    # a,b,r2 = lf.update_fit()    
    # sub.plot(xs, a+xs*b, c='orange')
    # print "err", err(a,b,xs,ys)

    plt.savefig(pp, format='pdf')
    pp.close()
