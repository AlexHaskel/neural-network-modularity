import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import product
from sklearn.metrics import mutual_info_score
from utils.information_metrics import total_correlation

def mean_absolute_correlation_score(X):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features)")

    if X.shape[0] < 2:
        return np.nan

    correlation_matrix = np.corrcoef(X, rowvar=False)
    offdiag = correlation_matrix[
        np.triu_indices_from(correlation_matrix, k=1)
    ]

    if not np.all(np.isfinite(offdiag)):
        return np.nan

    return float(np.mean(np.abs(offdiag)))

def mean_binned_mi_score(X, bin_edges):
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

            labels_i = binned[:, i]
            labels_j = binned[:, j]

            mi = mutual_info_score(labels_i, labels_j)

            #Calculate MI entries
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    offdiag = np.triu_indices(n_features, k=1)
    unique_offdiag_mi = mi_matrix[offdiag]

    return float(np.nanmean(unique_offdiag_mi))

def make_mean_binned_mi_score(bin_edges):
    bin_edges = np.asarray(bin_edges, dtype=float)

    def score(X):
        return mean_binned_mi_score(X, bin_edges)

    return score

def make_total_correlation_score(bin_edges):
    bin_edges = np.asarray(bin_edges, dtype=float)

    def score(X):
        X = np.asarray(X)

        binned_X = np.column_stack([
            np.digitize(
                X[:, j],
                bin_edges[1:-1],
            )
            for j in range(X.shape[1])
        ])

        return total_correlation(binned_X)

    return score       

# Generate a dataset by greedy candidate selection.
# At each iteration:
    # 1. Draw `num_candidates` iid candidate points.
    # 2. Add each candidate temporarily to the current dataset.
    # 3. Evaluate `score_function` on each proposed dataset.
    # 4. Permanently add a candidate achieving the minimum score.

# Parameters
# score_function:
#     Function accepting an array of shape (n_samples, n_features) 
#     and returning one scalar score. Lower scores are preferred.

# num_candidates:
#     Number of candidate points evaluated at each iteration.
#     num_candidates=1 reduces to ordinary sequential iid sampling.

# initial_size:
#     Size of the initial iid seed dataset.

# lower, upper:
#     Coordinate sampling bounds.

# return_history:
#     If True, also return the selected objective score at each
#     iteration.
def generate_greedy_metric_dataset(
    num_samples,
    n_features,
    rng,
    score_function,
    num_candidates,
    initial_size,
    lower,
    upper,
    return_history = False,
    ):
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    if n_features < 1:
        raise ValueError("n_features must be at least 1")

    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1")

    if initial_size < 1:
        raise ValueError("initial_size must be at least 1")

    current_X = rng.uniform(
        low=lower,
        high=upper,
        size=(initial_size, n_features),
    )

    history = []
    if return_history:
        # Optionally record the objective on the initial dataset.
        try:
            initial_score = float(score_function(current_X))
        except (ValueError, FloatingPointError):
            initial_score = np.nan

        history.append({
            "n": len(current_X),
            "score": initial_score,
        })

    while len(current_X) < num_samples:
        candidates = rng.uniform(
            low=lower,
            high=upper,
            size=(num_candidates, n_features),
        )

        scores = np.full(num_candidates, np.nan, dtype=float)

        for candidate_index, candidate in enumerate(candidates):
            proposed_X = np.vstack((current_X, candidate))
            score = float(score_function(proposed_X))

            if not np.isfinite(score):
                raise RuntimeError(f"Non-finite score for candidate {candidate_index}: {score}")

            scores[candidate_index] = score

        best_score = np.min(scores)

        # Random tie-breaking is useful for discrete/binned objectives.
        best_indices = np.flatnonzero(
            np.isclose(
                scores,
                best_score,
                rtol=1e-10,
                atol=1e-12,
            )
        )

        selected_index = int(rng.choice(best_indices))
        selected_candidate = candidates[selected_index]

        current_X = np.vstack((current_X, selected_candidate))

        if return_history:
            history.append({
                "n": len(current_X),
                "score": best_score,
                "candidate_index": selected_index,
                "num_tied_best": len(best_indices),
                "candidate_score_min": float(np.min(scores)),
                "candidate_score_max": float(np.max(scores)),
                "candidate_score_std": float(np.std(scores)),
            })

    current_X = current_X.astype(np.float32)

    if return_history:
        return current_X, history

    return current_X


# Dataset Definition
class GeneralDataset(Dataset):
    def __init__(self,
                 sampling_mode = "random_data",
                 seed=None,
                 encode_num = 4,
                 num_samples = 1000,
                 mean = 0,
                 std_dev = 2,
                 mapfun = lambda x : x,
                 modWise = False,
                 shuffle_frac = 0.5,
                 jitter_frac = 0.0,
                 iid_frac = 0.0,
                 num_grids=4,
                 num_removed=0,
                 add_iid=0,
                 mi_mod_levels = 3,
                 mc_objective = None,
                 mc_num_candidates = None,
                 mc_initial_size = None,
                 mc_return_history=False,
                 grid_center = 0,
                 grid_spacing = 2,
                 custom_centers = None,
                 custom_spacings = None,
                 custom_grid_levels = None,
                 ):

        self.mc_history = None
        
        #######################################
        ### Draw from uniform distributions ###
        #######################################
        if sampling_mode == "random_data":
            rng = np.random.default_rng(seed)
            self.data = (rng.random((num_samples, encode_num)) - 0.5) * 2 * std_dev + mean
        
        #########################################################
        ### Evenly sampled over whole range and permuted data ###
        #########################################################
        elif sampling_mode == "permuted":
            col = np.linspace(start=mean-std_dev, stop=mean+std_dev, num=num_samples, endpoint=False)
            X = np.column_stack([col]*encode_num)
            
            if shuffle_frac > 0:
                rng = np.random.default_rng()
                num_samples, encode_num = X.shape
                k = int(round(shuffle_frac * num_samples))

                for j in range(1, encode_num):
                    rows = rng.choice(num_samples, size=k, replace=False)
                    shuffled_values = X[rows, j].copy()
                    rng.shuffle(shuffled_values)
                    X[rows, j] = shuffled_values

            self.data = X


        #############################################
        ### Data samples in grid-like arrangement ###
        #############################################
        elif sampling_mode == "grid":
            rng = np.random.default_rng(seed)
            #take the qth root of the number of samples
            q = round(num_samples ** (1 / encode_num))

            #Validate the sample size to make sure it is an (encode_num)th power
            if q ** encode_num != num_samples:
                raise ValueError(
                f"num_samples={num_samples} is not a perfect "
                f"{encode_num}th power")
            
            #Create q evenly spaced points
            col = np.linspace(start=mean-std_dev, stop=mean+std_dev, num=q, dtype=np.float32)
            
            #Take cartesian products with self encode_num times
            X = np.array(list(product(col, repeat=encode_num)),dtype=np.float32)

            grid_spacing = col[1] - col[0]
            jitter_radius = jitter_frac * grid_spacing

            noise = rng.uniform(
                -jitter_radius,
                jitter_radius,
                size=X.shape,
            )

            X_jittered = X + noise

            self.data = X_jittered

        ##############################################################
        ### Data partially in grid-like arrangement, partially iid ###
        ##############################################################
        elif sampling_mode == "grid_partial":
            rng = np.random.default_rng(seed)
            #take the qth root of the number of samples
            q = round(num_samples ** (1 / encode_num))

            #Validate the sample size to make sure it is an (encode_num)th power
            if q ** encode_num != num_samples:
                raise ValueError(
                f"num_samples={num_samples} is not a perfect "
                f"{encode_num}th power")
            
            #Create q evenly spaced points
            col = np.linspace(start=mean-std_dev, stop=mean+std_dev, num=q, dtype=np.float32)
            
            #Take cartesian products with self encode_num times
            X = np.array(list(product(col, repeat=encode_num)),dtype=np.float32)

            #Portion that will be in grid and portion that will be iid
            n_iid = int(round(iid_frac * num_samples))
            n_grid = num_samples - n_iid

            # X has shape (num_samples, encode_num)
            # Choose indexes that will be preserved as grid
            keep_indices = rng.choice(
                len(X),
                size=n_grid,
                replace=False,
            )

            X_retained = X[keep_indices]

            # Generate an array of iid samples
            X_iid = (
                rng.random((n_iid, encode_num)) - 0.5
            ) * 2 * std_dev + mean

            data = np.concatenate(
                [X_retained, X_iid],
                axis=0,
            ).astype(np.float32)

            remove_indices = rng.choice(
                len(data),
                size=num_removed,
                replace=False,
            )

            data = np.delete(data, remove_indices, axis=0)
            print(f"total after deletion: {len(data)} points")
            self.data = data[rng.permutation(len(data))]

        ############################################################
        ### Data in grid-like arrangement, with added iid points ###
        ############################################################
        elif sampling_mode == "grid_additions":
            rng = np.random.default_rng(seed)
            #take the qth root of the number of samples
            q = round(num_samples ** (1 / encode_num))

            #Validate the sample size to make sure it is an (encode_num)th power
            if q ** encode_num != num_samples:
                raise ValueError(
                f"num_samples={num_samples} is not a perfect "
                f"{encode_num}th power")
            
            #Create q evenly spaced points
            col = np.linspace(start=mean-std_dev, stop=mean+std_dev, num=q, dtype=np.float32)
            
            #Take cartesian products with self encode_num times
            X = np.array(list(product(col, repeat=encode_num)),dtype=np.float32)

            # Generate an array of iid samples
            X_iid = (
                rng.random((add_iid, encode_num)) - 0.5
            ) * 2 * std_dev + mean

            data = np.concatenate(
                [X, X_iid],
                axis=0,
            ).astype(np.float32)

            print(f"Random points generated: {X_iid}\n")

            print(
                    f"Grid points: {len(X)}, "
                    f"added iid points: {len(X_iid)}, "
                    f"total points: {len(data)}"
                )
            self.data = data[rng.permutation(len(data))]

        ##############################
        ### Base points + variants ###
        ##############################
        # Repeatedly, a "base point" (x1,x2,x3,x4) is generated, followed by:  
        # (a,x2,x3,x4), (x1,b,x3,x4), (x1,x2,c,x4), (x1,x2,x3,d).  
        elif sampling_mode == "base_points":
            data = []
            group_size = encode_num + 1  # base + one variant per coordinate
            rng = np.random.default_rng(seed)

            while len(data) + group_size <= num_samples:
                #Generate a "base point"
                base = rng.uniform(
                    mean - std_dev,
                    mean + std_dev,
                    size=encode_num,
                )
                data.append(base.copy())

                #Pick (encode_num) random values
                replacements = rng.uniform(
                    mean - std_dev,
                    mean + std_dev,
                    size=encode_num,
                )

                #For each random value, replace each coordinate of the base point, one at a time.
                for j in range(encode_num):
                    variant = base.copy()
                    variant[j] = replacements[j]
                    data.append(variant)

            remaining = num_samples - len(data)

            # If there are points remaining, fill remainder slots with random points
            if remaining > 0:
                extra_points = rng.uniform(
                    mean - std_dev,
                    mean + std_dev,
                    size=(remaining, encode_num),
                )
                data.extend(extra_points)

            data = np.asarray(data, dtype=np.float32)
            data = data[rng.permutation(len(data))]            
            self.data = data

        #############################################
        ### Data samples in sub-grids, diagonally ###
        #############################################
        elif sampling_mode == "sub_grid_diag":
            rng = np.random.default_rng(seed)

            if num_samples % num_grids != 0:
                raise ValueError(
                    f"num_samples={num_samples} must be divisible by "
                    f"num_grids={num_grids}"
                )

            data = []

            subgrid_size = num_samples//num_grids
            #Endpoints
            endpoints = np.linspace(start=mean-std_dev, stop=mean+std_dev, num=num_grids+1, dtype=np.float32)

            #take the qth root of the number of samples
            q = round(subgrid_size ** (1 / encode_num))

            #Validate the sample size to make sure it is an (encode_num)th power
            if q ** encode_num != subgrid_size:
                raise ValueError(
                f"subgrid_size={subgrid_size} is not a perfect "
                f"{encode_num}th power")

            for i in range(num_grids):
                #Create q evenly spaced points
                col = np.linspace(start=endpoints[i], stop=endpoints[i+1], num=q, dtype=np.float32)

                #Take cartesian products with self encode_num times
                X = np.array(list(product(col, repeat=encode_num)),dtype=np.float32)

                data.extend(X)

            data = np.asarray(data, dtype=np.float32)
            data = data[rng.permutation(len(data))]
            self.data = data

        ##################################
        ### Grids as concentric shells ###
        ##################################
        elif sampling_mode == "concentric_shells":
            rng = np.random.default_rng(seed)

            data = []

            radii = np.arange(
                std_dev,
                0.0,
                -std_dev / num_grids,
                dtype=np.float32,
            )

            for radius in radii:
                levels = np.array([-radius, radius], dtype=np.float32)

                X = np.array(
                    list(product(levels, repeat=encode_num)),
                    dtype=np.float32,
                )

                data.extend(X)

            data = np.asarray(data, dtype=np.float32)
            data = data[rng.permutation(len(data))]
            self.data = data
            
        ##########################################
        ### Data permuted under some threshold ###
        ##########################################
        elif sampling_mode == "smart_permuted":
            col = np.linspace(start=mean-std_dev, stop=mean+std_dev, num=num_samples, endpoint=False)
            X = np.column_stack([col]*encode_num)
            
            def corr_stats(X):
                C = np.corrcoef(X, rowvar=False)
                off = C[np.triu_indices_from(C, k=1)]
                return {
                    "mean_abs_corr": np.mean(np.abs(off)),
                    "max_abs_corr": np.max(np.abs(off)),
                    "corr_matrix": C,
                }

            rng = np.random.default_rng()
            num_samples, encode_num = X.shape
            k = num_samples

            for j in range(1, encode_num):
                rows = rng.choice(num_samples, size=k, replace=False)
                shuffled_values = X[rows, j].copy()
                rng.shuffle(shuffled_values)
                X[rows, j] = shuffled_values

            self.data = X

        ################################################################
        ### Pairwise Mutual Information Adversarial Modular Datasets ###
        ################################################################
        elif sampling_mode == "mi_modular":
            def linear_mod_dataset(q, coefficient_pairs=None):
                """
                Generate a modular dataset from two latent variables a and b.
                Each coordinate is:
                    X_j = (c_j * a + d_j * b) mod q
                Returns integer labels in {0, ..., q - 1}.
                """
                if coefficient_pairs is None:
                    coefficient_pairs = [
                        (1, 0),  # X1 = a
                        (0, 1),  # X2 = b
                        (1, 1),  # X3 = a + b
                        (1, 2),  # X4 = a + 2b
                    ]

                latent_pairs = np.asarray(
                    list(product(range(q), repeat=2)),
                    dtype=int,
                )

                a = latent_pairs[:, 0]
                b = latent_pairs[:, 1]

                columns = [
                    (c * a + d * b) % q
                    for c, d in coefficient_pairs
                ]

                return np.column_stack(columns)

            def map_levels_to_interval(X_labels, q, lower=-2.0, upper=2.0):
                #Map the dataset generated by linear_mod_dataset
                #into a given range.

                levels = np.linspace(lower, upper, q)
                return levels[X_labels].astype(np.float32)
                

            # 1. Perform modular arithmetic on integer labels
            dataset_labels = linear_mod_dataset(mi_mod_levels)

            # 2. Map the finished labels into [-2, 2]
            dataset = map_levels_to_interval(dataset_labels, mi_mod_levels)
            
            print(f"Generated dataset with {len(dataset)} samples.")
            
            self.data = dataset
        
        ###################################################################
        ### Draw with Monte Carlo strategy to minimize a dataset metric ###
        ###################################################################
        elif sampling_mode == "monte_carlo":
            rng = np.random.default_rng(seed)

            dataset = generate_greedy_metric_dataset(
            num_samples = num_samples,
            n_features = encode_num,
            rng = rng,
            score_function = mc_objective,
            num_candidates = mc_num_candidates,
            initial_size = mc_initial_size,
            lower = mean - std_dev,
            upper = mean + std_dev,
            return_history = mc_return_history,
            )
            
            
            if mc_return_history:
                self.data, self.mc_history = dataset
            else:
                self.data = dataset
                self.mc_history = None

        ############################################################
        ### Create customized grid with custom center and offset ###
        ############################################################
        elif sampling_mode == "customized_grid":
            if custom_grid_levels is None:
                if custom_centers is None:
                    centers = np.full(encode_num, grid_center, dtype=float)
                else:
                    centers = np.asarray(custom_centers, dtype=float)

                if custom_spacings is None:
                    spacings = np.full(encode_num, grid_spacing, dtype=float)
                else:
                    spacings = np.asarray(custom_spacings, dtype=float)

                coordinate_levels = [
                    np.array([c - d, c, c + d])
                    for c, d in zip(centers, spacings)
                    ]
            else:
                coordinate_levels = [
                    np.asarray(levels, dtype=float)
                    for levels in custom_grid_levels
                    ]

            self.data = np.asarray(
                list(product(*coordinate_levels)),
                dtype=float,
            )
            
        #################################
        ### modWise data distribution ###
        #################################
        if modWise:
            x = np.zeros((num_samples, encode_num))
            random_indices = np.random.randint(0, encode_num, (num_samples,))  # A random index for each row
            x[np.arange(num_samples), random_indices] = self.data[np.arange(num_samples), random_indices]
            self.data = x
        
        ########################
        ### Generate outputs ###
        ########################

        self.num_samples = len(self.data)
        self.mapfun = mapfun
        self.__generate__()

    def __generate__(self):
        self.targets = []
        for i in range(self.num_samples):
            self.targets.append(self.mapfun(self.data[i]))
        self.targets = np.array(self.targets)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

    def get_all_data(self):
        # Return all data and targets as tensors
        inputs = torch.tensor(self.data, dtype=torch.float32)
        outputs = torch.tensor(self.targets, dtype=torch.float32)
        return inputs, outputs
    
   
