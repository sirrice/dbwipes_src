from sqlalchemy import *
import matplotlib
from matplotlib.patches import Rectangle as Rect 
from matplotlib.backends.backend_pdf import PdfPages 
import matplotlib.pyplot as plt 
from matplotlib import cm 
matplotlib.use("Agg") 


def get_all_ids(where):
    db = create_engine('postgresql://localhost/dbwipes')
    with db.begin() as conn:
        q = """select f1, ids from stats where %s""" % where
        for f1, idsstr in conn.execute(q):
            if idsstr:
                yield f1, set(map(int, map(float, idsstr.split(","))))
    db.dispose()

def get_pts(ids=None):
    db = create_engine('postgresql://localhost/sigmod')
    with db.begin() as conn:
        q = """select id, a_0, a_1, v/100. from data_2_2_1000_0d50_30uo where g >= 8""" 
        ret = []
        for id,x,y,v in conn.execute(q).fetchall():
            if not ids or id in ids:
                ret.append((x,y,v))
    db.dispose()
    return ret

def get_bounds(pts):
    if not pts:
        return (0,0), (0,0)
    xs, ys, _ = zip(*pts)
    return (min(xs), min(ys)), (max(xs)-min(xs), max(ys)-min(ys))


pp = PdfPages('points.pdf')
fig = plt.figure(figsize=(15,15))
sub = fig.add_subplot(111)


xs,ys,vs = zip(*get_pts())
sub.scatter(xs,ys,c=vs, alpha=0.6, lw=0)

all_bounds = []
for f1, ids in get_all_ids('expid = 85 and c = .5  and f1 > 0.05'):
    bounds = get_bounds(get_pts(ids))
    c = cm.jet(f1)
    r = Rect(bounds[0], bounds[1][0], bounds[1][1], alpha=0.85, ec=c,  fill=False, lw=1.5) 
    sub.add_patch(r)

plt.savefig(pp, format='pdf') 

pp.close()

