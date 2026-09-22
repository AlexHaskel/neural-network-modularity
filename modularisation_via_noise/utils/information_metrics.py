import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics.cluster import adjusted_mutual_info_score
import warnings

def bin_occupancy_summary(binned):
    binned = np.asarray(binned)
    n_samples, n_features = binned.shape

    occupied_bins = np.array([
        len(np.unique(binned[:, j]))
        for j in range(n_features)
    ])

    samples_per_occupied_bin = n_samples / occupied_bins

    return {
        "mean_occupied_bins": float(np.mean(occupied_bins)),
        "max_occupied_bins": int(np.max(occupied_bins)),
        "min_occupied_bins": int(np.min(occupied_bins)),
        
        "mean_samples_per_occupied_bin": float(np.mean(samples_per_occupied_bin)),
        "max_samples_per_occupied_bin": float(np.max(samples_per_occupied_bin)),
        "min_samples_per_occupied_bin": float(np.min(samples_per_occupied_bin)),
    }

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


def empirical_mutual_info(labels_x, labels_y):
    labels_x = np.asarray(labels_x)
    labels_y = np.asarray(labels_y)

    if labels_x.ndim != 1 or labels_y.ndim != 1:
        raise ValueError("labels_x and labels_y must be one-dimensional")

    if labels_x.shape[0] != labels_y.shape[0]:
        raise ValueError("labels_x and labels_y must have the same length")

    h_x = empirical_entropy(labels_x)
    h_y = empirical_entropy(labels_y)
    h_xy = empirical_entropy(np.column_stack((labels_x, labels_y)))

    mi = h_x + h_y - h_xy

    # Small negative values can happen from floating point roundoff.
    return float(max(mi, 0.0))

def arithmetic_normalized_mutual_info(labels_x, labels_y):
    h_x = empirical_entropy(labels_x)
    h_y = empirical_entropy(labels_y)

    denom = h_x + h_y
    if denom == 0.0:
        return np.nan

    mi = empirical_mutual_info(labels_x, labels_y)

    return float( (2.0 * mi) / denom )


# Compute joint-entropy-normalized mutual information:
# NMI_joint(X, Y) = I(X; Y) / H(X, Y)
# Inputs must be one-dimensional arrays of discrete labels.
def joint_normalized_mutual_info(labels_x, labels_y):
    labels_x = np.asarray(labels_x)
    labels_y = np.asarray(labels_y)

    if labels_x.ndim != 1 or labels_y.ndim != 1:
        raise ValueError("labels_x and labels_y must be one-dimensional")

    if labels_x.shape[0] != labels_y.shape[0]:
        raise ValueError("labels_x and labels_y must have the same length")

    mi = empirical_mutual_info(labels_x, labels_y)

    joint_labels = np.column_stack((labels_x, labels_y))
    joint_entropy = empirical_entropy(joint_labels)

    # H(X, Y) = 0 only when the joint variable is constant.
    # In that case I(X; Y) / H(X, Y) is undefined.
    # The matrix-building function handles the diagonal convention separately.
    if joint_entropy == 0.0:
        print(f"Joint Entropy is null")
        return np.nan

    return float(mi / joint_entropy)


# Calculate the Total Correlation between coordinate columns.
# Total Correlation is a generalization multivariate version of Mutual Information.
# TC(X1, X2, ..., Xn) = [ H(X1) + ... + H(Xn) ] - H(X1, ..., Xn)
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


# Calculate a normalized version of Total Correlation.
# This is calculated by dividing by the sum of entropies,
# and then multiplying by dim/dim-1.
# This generalizes the calculation of normalized MI.
def normalized_total_correlation_sum(binned_X):
    binned_X = np.asarray(binned_X)

    if binned_X.ndim != 2:
        raise ValueError("binned_X must have shape (n_samples, n_features)")

    d = binned_X.shape[1]

    marginal_entropy_sum = sum(
        empirical_entropy(binned_X[:, j])
        for j in range(d)
    )

    if marginal_entropy_sum == 0.0:
        return np.nan

    tc = total_correlation(binned_X)

    return float(
        (d / (d - 1)) * (tc / marginal_entropy_sum)
    )


# Calulate binned mutual information and normalized binned mutual information
# The default scikit learn implementation for normalization uses 
# Normalized MI = 2*I(X_i; X_j) / H(X_i) + H(X_j)
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
    nmi_matrix = np.zeros((n_features, n_features))
    nmi_joint_matrix = np.zeros((n_features, n_features))
    adj_mi_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i, n_features):

            labels_i = binned[:, i]
            labels_j = binned[:, j]

            mi = mutual_info_score(labels_i, labels_j)
            nmi = normalized_mutual_info_score(labels_i, labels_j, average_method="arithmetic")
            if i == j:
                nmi_joint = 1.0
            else:
                nmi_joint = joint_normalized_mutual_info(labels_i, labels_j)               
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The number of unique classes is greater than 50% of the number of samples.",
                    category=UserWarning,
                )

                adj_mi = adjusted_mutual_info_score(labels_i,labels_j,average_method="arithmetic")               

            #Calculate MI entries
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

            #Calculate normalized MI entries
            nmi_matrix[i, j] = nmi
            nmi_matrix[j, i] = nmi

            #Calculate joint-entropy normalize MI entries
            nmi_joint_matrix[i, j] = nmi_joint
            nmi_joint_matrix[j, i] = nmi_joint

            #Calculate Adjusted for chance MI matrix entries
            adj_mi_matrix[i, j] = adj_mi
            adj_mi_matrix[j, i] = adj_mi
    
    offdiag = np.triu_indices(n_features, k=1)
    unique_offdiag_mi = mi_matrix[offdiag]
    unique_offdiag_norm_mi = nmi_matrix[offdiag]
    unique_offdiag_nmi_joint = nmi_joint_matrix[offdiag]
    unique_offdiag_adj_mi = adj_mi_matrix[offdiag]
    diagnostics = bin_occupancy_summary(binned)

    return {
        "mi_matrix": mi_matrix,
        "mi_mean": float(np.nanmean(unique_offdiag_mi)),
        "mi_max": float(np.nanmax(unique_offdiag_mi)),

        "nmi_arithm_matrix": nmi_matrix,
        "nmi_arithm_mean": float(np.nanmean(unique_offdiag_norm_mi)),
        "nmi_arithm_max": float(np.nanmax(unique_offdiag_norm_mi)),

        "nmi_joint_matrix": nmi_joint_matrix,
        "nmi_joint_mean": float(np.nanmean(unique_offdiag_nmi_joint)),
        "nmi_joint_max": float(np.nanmax(unique_offdiag_nmi_joint)),

        "adj_mi_matrix": adj_mi_matrix,
        "adj_mi_mean": float(np.nanmean(unique_offdiag_adj_mi)),
        "adj_mi_max": float(np.nanmax(unique_offdiag_adj_mi)),
        "adj_mi_min": float(np.nanmin(unique_offdiag_adj_mi)),
        
        "total_correlation": total_correlation(binned),
        "normalized_total_correlation": normalized_total_correlation_sum(binned),
        **diagnostics,
    }


def pairwise_ksg_mi_sklearn(X, n_neighbors=3, random_state=0):
    X = np.asarray(X, dtype=float)

    n_features = X.shape[1]
    mi_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            mi = mutual_info_regression(
                X[:, [i]],
                X[:, j],
                discrete_features=False,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )[0]

            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    values = mi_matrix[np.triu_indices(n_features, k=1)]

    return {
        "ksg_matrix": mi_matrix,
        "ksg_mean": float(values.mean()),
        "ksg_max": float(values.max()),
    }