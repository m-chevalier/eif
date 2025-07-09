import numpy as np

class iForest:
    
    def __init__(self, ntrees: int, sample_size: int, limit:int, extension_level:int, seed:int, parallel: bool):
        """
        iForest(X, ntrees,  sample_size, limit=None, extension_level=0, seed=-1)
        Initialize a forest by passing in training data, number of trees to be used and the subsample size.

        Parameters
        ----------
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

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

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

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        ...
