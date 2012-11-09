import math
import pdb
import random
import numpy as np
import orange
import Orange
import sys
sys.path.extend(['.', '..'])

from itertools import chain, product, groupby
from collections import defaultdict
from operator import sub

from util.prob import *
from bounding_box import *
from learners.cn2sd.rule import SDRule



class Cluster(object):
    _id = 0
    
    def __init__(self, bbox, error, cols, parents=[], discretes={}, **kwargs):
        """
        @param cols the set of continuous columns
        """
        self.error = error
        self.bbox = (tuple(bbox[0]), tuple(bbox[1]))
        self.discretes = defaultdict(set)
        self.centroid = tuple([np.mean((p1, p2)) for p1, p2 in zip(*self.bbox)])
        self.volume = volume(bbox)
        self.cols = cols
        self.parents = parents
        self.max_anc_error = self.max_ancestor_error()
        self.npts = kwargs.get('npts', 1)
        self.kwargs = kwargs
        self.id = Cluster._id


        self.bad_inf = None
        self.good_inf = None
        self.mean_inf = None
        self.idxs = []

        Cluster._id += 1

        for k,vs in discretes.iteritems():
            self.discretes[k].update(vs)

        self.hash = None
        self._bound_hash = None

    def clone(self):
        return Cluster(self.bbox, self.error, self.cols,
                       parents=self.parents, discretes=self.discretes, **self.kwargs)

    def max_ancestor_error(self):
        if not self.parents:
            return self.error
        return max(self.error, max(p.max_anc_error for p in self.parents))

    def split_on(self, c):
        ex_clusters = [] # clusters that don't intersect with c
        in_clusters = [] # clusters that intersect with c
        
        if not volume(intersection_box(self.bbox, c.bbox)):
            return ()

        if self.discretes:
            # they have same keys, just need to diff them
            myd = self.discretes
            od = c.discretes

            intersect_discretes = dict()
            excluded_discretes = dict()
            for key in self.discretes:
                myvals = set(myd[key])
                ovals = set(od[key])
                intersect_discretes[key] = myvals.intersection(ovals)
                excluded_discretes[key] = myvals.difference(ovals)

            if not sum(map(len, intersect_discretes.values())):
                return ()

            if not sum(map(len, excluded_discretes.values())):
                return ([self.clone()], [])


            excluded_cluster = self.clone() 
            excluded_cluster.discretes = excluded_discretes
            ex_clusters.append(excluded_cluster)
        else:
            intersect_discretes = dict(self.discretes)

        if box_contained(self.bbox, c.bbox):
            intersect_cluster = self.clone()
            intersect_cluster.discretes = intersect_discretes
            return ([intersect_cluster], ex_clusters)

        # how the painful part.. splitting the contiuous domain
        inner_mins = map(max, zip(self.bbox[0], c.bbox[0]))
        inner_maxs = map(min, zip(self.bbox[1], c.bbox[1]))
        all_ranges = [[] for i in inner_mins]
        for idx, (bmin, bmax, imin, imax) in enumerate(zip(self.bbox[0], self.bbox[1], inner_mins, inner_maxs)):
            vals = sorted(set([bmin, bmax, imin, imax]))
            for i in xrange(len(vals)-1):
                all_ranges[idx].append((vals[i], vals[i+1]))

        all_boxes = product(*all_ranges)
        groups = defaultdict(list)
        for box in all_boxes:
            new_cluster = self.clone()
            new_cluster.bbox = tuple(zip(*box))
            new_cluster.discretes = dict(intersect_discretes)
            groups[box_contained(new_cluster.bbox, c.bbox)].append(new_cluster)

        ex_clusters.extend(groups.get(False, []))
        in_clusters = groups.get(True, [])
        for ec in ex_clusters:
            ec.parents = (self, c)
        return (list(in_clusters), list(ex_clusters))


    @staticmethod
    def merge(c1, c2, intersecting_clusters, min_volume):
        if c1.cols != c2.cols:
            raise RuntimeError("columns do not match!  %s vs %s" % (str(c1.cols), str(c2.cols)))
        
        newbbox = bounding_box(c1.bbox, c2.bbox)

        weights = []
        errors = []
        for intersection in intersecting_clusters:
            bbox = intersection_box(intersection.bbox, newbbox)
            weights.append(volume(bbox) or min_volume)
            errors.append(intersection.error)

        total_weight = sum(weights) or 1.
        if not total_weight:
            return None

        discretes = defaultdict(set)
        for k in chain(c1.discretes.keys(), c2.discretes.keys()):
            discretes[k].update(c1.discretes.get(k, ()))
            discretes[k].update(c2.discretes.get(k, ()))

        newerror = sum(e*w for e,w in zip(errors, weights)) / total_weight
        return Cluster(newbbox, newerror, c1.cols, parents=(c1, c2), discretes=discretes)


    @staticmethod
    def from_rule(rule, cols, error=None, parents=None):
        """
        @param cols list of attribute names
        """
        domain = rule.data.domain
        name_to_bounds = {}
        discretes = {}
        
        for c in rule.filter.conditions:
            attr = domain[c.position]
            name = attr.name
            if attr.varType == orange.VarTypes.Discrete:
                if name in cols:
                    discretes[name] = [int(v) for v in c.values]
            else:
                name_to_bounds[name] = (c.min, c.max)

        mins, maxs = [], []
        cont_cols = []
        for col in cols:
            if col in name_to_bounds:
                mins.append(name_to_bounds[col][0])
                maxs.append(name_to_bounds[col][1])
                cont_cols.append(col)
            
        bbox = (tuple(mins), tuple(maxs))
        error = error or rule.quality
        return Cluster(bbox, error, cont_cols, parents=parents, discretes=discretes, npts=len(rule.examples))

    def adjacent(self, o, thresh=0.7):
        """@return True if they overlap on one or more continuous attribute, 
           and their discrete attributes intersect"""
        d_intersects =  self.discretes_intersect(o)
        if not d_intersects:
            return False

        d_same = self.discretes_same(o)
        intersection = intersection_box(self.bbox, o.bbox)
        make_diff = lambda bbox: map(lambda p:p[1]-p[0], zip(*bbox))
        diffs = make_diff(intersection) 
        mydiffs = make_diff(self.bbox)
        odiffs = make_diff(o.bbox)

        overlapping = [d for d, md, od in zip(diffs, mydiffs, odiffs) if d > 0 and d/max(md,od) > thresh]
        if len(overlapping) > 0:
            return True
        return False


        if False and d_same:
            return len(filter(lambda d: d < 0, diffs)) > 0
        return volume(intersection) >= 0.7 * min(self.volume, o.volume)

    def contains(self, o, epsilon=0.):
        for key in o.discretes.keys():
            if key in self.discretes:
                diff = set(o.discretes[key]).difference(self.discretes[key])
                if len(diff):
                    return False
        return box_contained(o.bbox, self.bbox, epsilon=epsilon)

    def same(self, o, epsilon=0.):
        for key in o.discretes.keys():
            if key in self.discretes:
                diff = set(o.discretes[key]).difference(self.discretes[key])
                if len(diff):
                    return False

        return box_same(o.bbox, self.bbox, epsilon=epsilon)



    def discretes_same(self, o):
        mykeys = set(self.discretes.keys())
        okeys = set(o.discretes.keys())
        if mykeys != okeys:
            return False
        for key in mykeys:
            if set(self.discretes[key]) != set(o.discretes[key]):
                return False
        return True

    def discretes_intersect(self, o):
        myd = self.discretes
        od = o.discretes
        keys = set(myd.keys()).intersection(set(od.keys()))
        
        for key in keys:
            if len(od[key].intersection(myd[key])) < min(len(od[key]), len(myd[key]))-1:
                return False
        return True

    def to_rule(self, table, cols, cont_dists=None, disc_dists=None):
        """
        @param cols list of attribute names
        """
        domain = table.domain
        attrnames = [attr.name for attr in domain]
        cont_dists = cont_dists or dict(zip(attrnames, Orange.statistics.basic.Domain(table)))
        disc_dists = disc_dists or dict(zip(attrnames, Orange.statistics.distribution.Domain(table)))
        conds = []

        for col, bound in zip(self.cols, zip(*self.bbox)):
            attr = domain[col]
            pos = domain.index(attr)
            table_bound = cont_dists[attr.name]
            minv, maxv = max(table_bound.min, bound[0]), min(table_bound.max, bound[1])
            if maxv - minv > 0.9 * (table_bound.max-table_bound.min):
                continue
            
            conds.append(
                orange.ValueFilter_continuous(
                    position=pos,
                    max=bound[1],
                    min=bound[0]
                )
            )

        for disc_name, vidxs in self.discretes.iteritems():
            attr = domain[disc_name]
            disc_pos = domain.index(attr)
            vals = [orange.Value(attr, attr.values[int(vidx)]) for vidx in vidxs]

            if not vals or len(vals) == len(disc_dists[attr.name]):
                continue
            
            conds.append(
                orange.ValueFilter_discrete(
                    position = disc_pos,
                    values=vals
                )
            )

        rule = SDRule(table, None, conditions=conds)
        rule.quality = rule.score = self.error
        return rule

    def project(self, cols):
        """
        @param cols all of the columns (continuous and discrete) to keep
        """
        newbbox = [[], []]
        newcols = []
        for col, minv, maxv in zip(self.cols, self.bbox[0], self.bbox[1]):
            if col in cols:
                newbbox[0].append(minv)
                newbbox[1].append(maxv)
                newcols.append(col)

        newdiscretes = dict(filter(lambda (k,v): k in cols, self.discretes.iteritems()))
        return Cluster(newbbox, self.error, newcols, parents=self.parents, discretes=newdiscretes)



    def __ancestors__(self):
        if not self.parents:
            return (self,)
        ret = []
        for p in self.parents:
            ret.extend(p.ancestors)
        return ret
    ancestors = property(__ancestors__)
    
    def __str__(self):
        fmt = '\t'.join(['%.4f'] * len(self.bbox[0]))
        args = (self.error, fmt % self.bbox[0], fmt % self.bbox[1], str(self.discretes))
        return '%.6f\t%s\t%s\t%s' % args

    def __hash__(self):
        if self.hash is None:
            self.hash = hash((self.error, self.bbox, str(self.discretes)))
        return self.hash

    @property
    def bound_hash(self):
        if self._bound_hash is None:
            self._bound_hash = hash((self.bbox, str(self.discretes)))
        return self._bound_hash

    def __eq__(self, o):
        return hash(self) == hash(o)



def compute_clusters_threshold(clusters, nstds=1.):
    errors = [c.error for c in clusters]
    npts = [c.npts for c in clusters]
    #npts = [1] * len(clusters)
    mean, std = wmean(errors, npts), wstd(errors, npts)        
    thresh = max(0, min(max(errors), mean + nstds * std))
    return thresh


def filter_top_clusters(clusters, nstds=1.):
    """
    compute mean and std of error of clusters.  return clusters
    nstds above mean
    """
    if len(clusters) <= 1:
        return clusters
    thresh = compute_clusters_threshold(clusters, nstds)
    f = lambda c: c.error >= thresh
    return filter(f, clusters)

    errors = [c.error for c in clusters]
    minv, maxv, mean, std = min(errors), max(errors), np.mean(errors), np.std(errors)
    thresh = min(maxv, mean + nstds * std)
    thresh = 0.6 * (maxv - minv) + minv
    f = lambda c: c.error >= thresh
    return filter(f, clusters)


def normalize_cluster_errors(clusters):
    if not clusters:
        return clusters
    errors = [c.error for c in clusters]
    mine, maxe = min(errors), max(errors)
    div = 1. if maxe == mine else (maxe-mine)

    for c in clusters:
        c.error = (c.error - mine) / div
    return clusters


def clusters_to_rules(clusters, cols, table):
    import Orange
    attrnames = [attr.name for attr in table.domain]
    cont_dists = dict(zip(attrnames, Orange.statistics.basic.Domain(table)))
    disc_dists = dict(zip(attrnames, Orange.statistics.distribution.Domain(table)))
    args = {'cont_dists' : cont_dists,
            'disc_dists' : disc_dists}

    rules = [c.to_rule(table, cols, **args) for c in clusters]

    return rules
