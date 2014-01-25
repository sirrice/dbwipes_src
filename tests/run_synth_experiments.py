from test import run

def run_synth(db, pp, ndim, volperc):
  params = []
  for kdim in xrange(1, ndim+1):
    for uo in (30, 80):
      tablename = 'data_%d_%d_1000_0d%d_%duo' % (ndim, kdim, int(100.*volperc), uo)
      params.append((tablename, (2000, ndim, kdim, volperc, 10, 10, uo, 10)))

  for tablename, args in params:

    if ndim == 2 and kdim == 1:
        continue

    cs = [0., .25, .5, .75, 1.]
    run(db, tablename, pp, klasses=[Naive], max_wait=2*60, c=cs, granularity=20, naive=True, notes=tablename)
    run(db, tablename, pp, klasses=[NDT],  c=cs, granularity=20, naive=True, notes=tablename)
    run(db, tablename, pp, klasses= [BDT], l=[0.5], c=cs, epsilon=[0.0001, 0.001], tau=[0.1, 0.55], p=0.7, notes=tablename)
    run(db, tablename, pp, klasses= [BDT], l=[0.5], c=[1.], epsilon=[0.01, 0.1], tau=[0.1, 0.55], p=0.7, notes=tablename)
    run(db, tablename, pp, klasses=[MR], l=[.5], c=cs, granularity=20, notes=tablename)

if __name__ == '__main__':
  from sqlalchemy import *
  db = create_engine('postgresql://localhost/sigmod')
  pp = PdfPages('figs/test.pdf')

  print "did you run setup_experiments.py to generate the testing env?"

  run_synth(db, pp, 2, 0.5)
  run_synth(db, pp, 3, 0.5)
  run_synth(db, pp, 4, 0.5)

  pp.close()


