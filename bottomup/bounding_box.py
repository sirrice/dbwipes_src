from operator import mul, and_


def points_bounding_box(points):
    if not len(points):
        return ((0,), (0,))
    return (tuple(points.min(0)), tuple(points.max(0)))


def bounding_box(bbox1, bbox2):
    mins = tuple([min(min1, min2) for min1, min2 in zip(bbox1[0], bbox2[0])])
    maxs = tuple([max(max1, max2) for max1, max2 in zip(bbox1[1], bbox2[1])])
    return (mins, maxs)

def intersection_box(bbox1, bbox2):
    mins = tuple([max(min1, min2) for min1, min2 in zip(bbox1[0], bbox2[0])])
    maxs = tuple([min(max1, max2) for max1, max2 in zip(bbox1[1], bbox2[1])])
    return (mins, maxs)

def box_contained(box, bound):
    if not len(zip(*box)) or not len(zip(*bound)):
        return False
    return (reduce(and_, (min1 >= min2 for min1, min2 in zip(box[0], bound[0]))) and
            reduce(and_, (max1 <= max2 for max1, max2 in zip(box[1], bound[1]))) )

def volume(bbox):
    if not len(bbox[0]): return 0.
    return reduce(mul, (maxv - minv for minv, maxv in zip(*bbox)))

