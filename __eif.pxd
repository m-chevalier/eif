cdef extern from "eif.hxx":
    cdef cppclass iForest:
        int limit
        iForest (int, int, int, int, int)
        void fit (double*, int, int)
        void predict_non_parallel (double*, double*, int)
        void predict_parallel (double*, double*, int)
        void predictSingleTree (double*, double*, int, int)
        void OutputTreeNodes (int)
