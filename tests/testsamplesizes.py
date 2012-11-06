if __name__ == '__main__':
    import sys
    sys.path.extend( ['.', '..'])
    from util import *
    import numpy as np

    import matplotlib
    matplotlib.use("Agg")            
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('.')


    
    # pp = PdfPages('./figs/samplesizes.pdf')
    # fig = plt.figure(figsize=(20, 15))
    # sub = fig.add_subplot(111)
    # sub2 = sub.twinx()
    
    # for errprob in [0.0005, 0.001, 0.005, 0.01, 0.1]:
    #     xs, ys, ys2 = [], [], []

    #     for n in [10, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000,1000000]:
    #         xs.append( n )
    #         ys.append( best_sample_size(n, errprob) )
    #         ys2.append( ys[-1] / float(n) )

    #     sub.plot(xs, ys, label='%.4f error' % errprob)
    #     sub2.plot(xs, ys2, label='%.4f error' % errprob, alpha=0.5)        
    # sub.set_ylabel('min sample size for 95% conf')
    # sub2.set_ylabel('% sample size')
    # sub2.set_yscale('log')
    # sub.set_xscale('log')
    # sub.set_xlabel('population size')
    # sub.legend()
    # plt.savefig(pp, format='pdf')
    # pp.close()


    pp = PdfPages('./figs/samplefit.pdf')
    import scipy.optimize as optimize
    from math import log
    xs, ys = [], []
    for n in [10, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000,1000000]:
        xs.append( n )
        ys.append( best_sample_size(n, 0.001) / float(n))
    xs, ys = np.array(xs), np.array(ys)
    def f(xs, a, b, c):
        return a * (xs**b)
    from scipy.optimize import curve_fit
    (a,b,c), pcov = curve_fit(f, xs, ys)
    print a,b,c

    fig = plt.figure(figsize=(20, 15))
    sub = fig.add_subplot(111)
    sub.plot(xs, a * xs**b-0.2)
    sub.scatter(xs, ys, s=15, lw=0)
    sub.scatter(xs, map(math.log, ys), c='red', s=15, lw=0)
    
    plt.savefig(pp, format='pdf')
    pp.close()



