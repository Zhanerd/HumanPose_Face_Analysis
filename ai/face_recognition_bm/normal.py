import numpy as np

def normalize(X, norm='l2', axis=1):
    """
    Normalize the dataset X so that each sample has norm 1.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.

    norm : ‘l1’, ‘l2’, or ‘max’, optional (‘l2’ by default)
        The norm to use to normalize each non zero sample (or each non-zero feature if axis is 0).

    axis : 0 or 1, optional (1 by default)
        Axis used to normalize the data along. If 1, independently normalize each sample,
        otherwise (if 0) normalize each feature.
    """
    if norm not in ('l1', 'l2', 'max'):
        raise ValueError("Invalid norm order. Only 'l1', 'l2', and 'max' norms are supported.")

    if axis not in (0, 1):
        raise ValueError("Invalid axis. Only 0 and 1 are supported.")

    if norm == 'l1':
        X_norm = np.abs(X).sum(axis=axis, keepdims=True)
    elif norm == 'l2':
        X_norm = np.sqrt((X ** 2).sum(axis=axis, keepdims=True))
    else:  # norm == 'max'
        X_norm = np.max(np.abs(X), axis=axis, keepdims=True)

    # Avoid division by zero
    X_norm[X_norm == 0] = 1

    return X / X_norm
