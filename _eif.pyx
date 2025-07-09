# Cython wrapper for Extended Isolation Forest

# distutils: language = c++
# distutils: sources  = eif.cxx
# cython: language_level = 3

import cython
import numpy as np
cimport numpy as np
from version import __version__
from libcpp cimport bool
from sklearn.utils import check_array

cimport __eif

np.import_array()

cdef class iForest:
    cdef int size_X
    cdef int dim
    cdef int _ntrees
    cdef int _limit
    cdef int sample
    cdef int tree_index
    cdef int exlevel
    cdef bool parallel_mode
    cdef __eif.iForest* thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__ (self, int ntrees, int sample_size, int limit=0, int extension_level=0, int seed=-1, bool parallel=True):  
        if extension_level < 0:
            raise Exception("Wrong Extension")
        self.thisptr = new __eif.iForest (ntrees, sample_size, limit, extension_level, seed, <bint> parallel)
        self.sample = sample_size
        self.parallel_mode = <bint> parallel
        self._ntrees = ntrees
        self._limit = self.thisptr.limit
        self.exlevel = extension_level

    @property
    def ntrees(self):
        return self._ntrees

    @property
    def limit(self):
        return self._limit

    def __dealloc__ (self):
        del self.thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def decision_function (self, X):
        cdef np.ndarray[double, ndim=1, mode="c"] S
        X = check_array(X)
        if not X.flags['C_CONTIGUOUS']:
            X = X.copy(order='C')
        S = np.empty(X.shape[0], dtype=np.float64, order='C')
        self.thisptr.predict(<double*> np.PyArray_DATA(S), <double*> np.PyArray_DATA(X), X.shape[0])
        return S

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_paths_single_tree (self, X, tree_index=0):
        cdef np.ndarray[double, ndim=1, mode="c"] S
        X = check_array(X)
        if not X.flags['C_CONTIGUOUS']:
            X = X.copy(order='C')
        S = np.empty(X.shape[0], dtype=np.float64, order='C')
        self.thisptr.predictSingleTree (<double*> np.PyArray_DATA(S), <double*> np.PyArray_DATA(X), X.shape[0], tree_index)
        return S

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict(self, X):
        X = check_array(X)
        if not X.flags['C_CONTIGUOUS']:
            X = X.copy(order='C')
        cdef np.ndarray[double, ndim=1, mode="c"] S = self.decision_function(X)
        return S

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, X, y=None):
        X = check_array(X)
        if not X.flags['C_CONTIGUOUS']:
            X = X.copy(order='C')
        self.size_X = X.shape[0]
        self.dim = X.shape[1]
        self.thisptr.fit (<double*> np.PyArray_DATA(X), self.size_X, self.dim)

    def output_tree_nodes (self, int tree_index):
        self.thisptr.OutputTreeNodes (tree_index)