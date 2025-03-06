import numpy as np

class iForest:
    
    def __init__(self, X: np.ndarray, ntrees: int, sample_size: int, limit:int, extension_level:int, seed:int, parallel: bool):
        """
        iForest(X, ntrees,  sample_size, limit=None, extension_level=0, seed=-1)
        Initialize a forest by passing in training data, number of trees to be used and the subsample size.

        Parameters
        ----------
        X : list of list of floats
            Training data. List of [x1,x2,...,xn] coordinate points.
        ntrees : int
            Number of trees to be used.
        sample_size : int
            The size of the subsample to be used in creation of each tree. Must be smaller than |X|
        limit : int
            The maximum allowed tree depth. This is by default set to average length of unsucessful search in a binary tree.
        extension_level : int
            Specifies degree of freedom in choosing the hyperplanes for dividing up data. Must be smaller than the dimension n of the dataset.
        seed : int
            Random seed for reproducibility.
        parallel : bool
            If True, the computation is done in parallel. If False, the computation is done in a single thread.
        """
        ...

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        decision_function(X)
        Compute anomaly scores for all data points in a dataset X

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        ...


    def compute_paths_single_tree(self, X_in, tree_index: int):
        ...

    def output_tree_nodes(self, tree_index: int):
        ...
