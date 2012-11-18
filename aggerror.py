import numpy as np
import Orange
from functions import *
from errfunc import *
from util import *

_logger = get_logger()


class AggErr(object):
    
    def __init__(self, agg, keys, npts, errtype, metadata={}):
        """
        @param npts either an int or a tuple (min, max)
        """
        self.agg = agg
        self.keys = keys
        self.npts = npts
        self.errtype = ErrTypes(errtype, metadata.get('erreq', None))
        self.erreq = metadata.get('erreq', None)
        self.metadata = metadata or {}

    def __str__(self):
        return str([str(self.agg), self.keys, self.npts, str(self.errtype), self.erreq])

    error_func = property(lambda self: self.__error_func__())

    def __error_func__(self):
        f = self.agg.func
        f.set_errtype(self.errtype)
        return f
        if self.agg.func == 'avg':
            return FastAvgErrFunc(self)
        if self.agg.func == 'stddev':
            return FastStdErrFunc(self)
        if self.agg.func == 'sum':
            return FastSumErrFunc(self)
        if self.agg.func == 'min':
            return MinErrFunc(self)
        if self.agg.func == 'corr':
            return FastCorrErrFunc(self)
        raise NotImplementedError
