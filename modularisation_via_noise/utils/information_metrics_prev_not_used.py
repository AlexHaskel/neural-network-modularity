import numpy as np
from sklearn.metrics import mutual_info_score

def empirical_entropy(labels):
    labels = np.asarray(labels)

    if labels.ndim == 1:
        _, counts = np.unique(labels, return_counts=True)
    elif labels.ndim == 2:
        _, counts = np.unique(labels, axis=0, return_counts=True)
    else:
        raise ValueError("labels must be one- or two-dimensional")

    probabilities = counts / counts.sum()

    return float(-np.sum(probabilities * np.log(probabilities)))

def total_correlation(binned_X):
    binned_X = np.asarray(binned_X)

    if binned_X.ndim != 2:
        raise ValueError("binned_X must have shape (n_samples, n_features)")
    
    marginal_entropy_sum = sum(
        empirical_entropy(binned_X[:, j])
        for j in range(binned_X.shape[1])
    )

    joint_entropy = empirical_entropy(binned_X)

    return float(marginal_entropy_sum - joint_entropy)


def binned_mi_summary(X, bin_edges):
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features)")
    
    n_features = X.shape[1]

    binned = np.column_stack([
        np.digitize(X[:, j], bin_edges[1:-1])
        for j in range(n_features)
    ])

    mi_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i, n_features):
            mi = mutual_info_score(binned[:, i], binned[:, j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    unique_offdiag = mi_matrix[np.triu_indices(n_features, k=1)]

    return {
        "matrix": mi_matrix,
        "mean": float(np.mean(unique_offdiag)),
        "max": float(np.max(unique_offdiag)),
        "total_correlation": total_correlation(binned),
    }