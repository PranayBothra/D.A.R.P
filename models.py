import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import numpy as np
from collections import Counter

class KNN:  # 1-KNN

    def __init__(self, k=3, distance="euclidean", task="classification"):
        # Validate the number of neighbors
        if k <= 0:
            raise ValueError('K should be a positive integer')

        # Validate the distance metric
        valid_distances = {"euclidean", "manhattan", "cosine"}
        if distance not in valid_distances:
            raise ValueError(f"Distance must be one of: {valid_distances}")

        # Validate the prediction task
        valid_tasks = {"classification", "regression"}
        if task not in valid_tasks:
            raise ValueError(f"Task must be one of: {valid_tasks}")

        self.k = k
        self.distance = distance
        self.task = task
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # Store the training data
        self.X_train = np.array(X_train)

        # Store target values
        self.y_train = np.array(y_train, dtype=float) if self.task == "regression" else np.array(y_train)

    def _compute_distances(self, X):
        # Compute pairwise Euclidean distances using matrix expansion for memory efficiency
        if self.distance == 'euclidean':
            X_sq = np.sum(X**2, axis=1)[:, None]
            train_sq = np.sum(self.X_train**2, axis=1)[None, :]
            dot_product = np.dot(X, self.X_train.T)
            
            sq_dists = X_sq + train_sq - 2 * dot_product
            return np.sqrt(np.maximum(sq_dists, 0.0))

        # Compute pairwise Manhattan distances
        elif self.distance == 'manhattan':
            return np.abs(X[:, None, :] - self.X_train[None, :, :]).sum(axis=2)

        # Compute cosine distance (1 - cosine similarity)
        elif self.distance == 'cosine':
            norm_X = np.linalg.norm(X, axis=1, keepdims=True)
            norm_train = np.linalg.norm(self.X_train, axis=1)
            denom = norm_X * norm_train
            dot = X @ self.X_train.T
            sim = np.where(denom == 0, 0.0, dot / np.where(denom == 0, 1, denom))
            return 1 - sim

    def predict(self, X):
        X = np.array(X)

        # Ensure the model has been trained
        if self.X_train is None:
            raise ValueError("Model must be fit before calling predict")

        # Validate k against the size of the training dataset
        if self.k > len(self.X_train):
            raise ValueError(
                f"k={self.k} cannot be greater than the number of training samples ({len(self.X_train)})"
            )

        # Accept a single sample by converting it to a 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Validate the input dimensions
        if X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D, got shape {X.shape}")

        # Ensure the number of features matches the training data
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fit on {self.X_train.shape[1]} features"
            )

        # Compute distances from each test sample to every training sample
        dists = self._compute_distances(X)

        # Find the indices of the k nearest neighbors using O(N) partitioning
        k_indices = np.argpartition(dists, self.k - 1, axis=1)[:, :self.k]

        # Sort only the extracted k neighbors in O(k log k) time to ensure correct tie-breaking
        row = np.arange(dists.shape[0])[:, None]
        order = np.argsort(dists[row, k_indices], axis=1)
        k_indices = k_indices[row, order]

        # Retrieve the corresponding target values
        k_nearest_values = self.y_train[k_indices]

        # Predict using the mean of the neighbors for regression
        if self.task == "regression":
            return k_nearest_values.mean(axis=1)

        # Predict using majority voting for classification
        predicted_labels = []
        for row_vals in k_nearest_values:
            counts = Counter(row_vals)
            max_votes = max(counts.values())

            # Resolve ties by selecting the label of the closest neighbor (now strictly accurate)
            tied = [label for label in row_vals if counts[label] == max_votes]
            predicted_labels.append(tied[0])

        return np.array(predicted_labels) 

class Kmeans: #2-Kmeans clustering
    def __init__(self, k=3, distance='euclidean', max_iters=300, tol=1e-3):
        if k <= 0:
            raise ValueError('Enter a valid number of clusters')
        
        valid_distances = {"euclidean", "manhattan"}
        if distance not in valid_distances:
            raise ValueError(f"Distance must be one of: {valid_distances}")
        
        self.k = k
        self.distance = distance
        self.max_iters = max_iters
        self.tol = tol
        self.centers = None
        self.labels_ = None
        self.sse_history_ = []

    def _seeds(self, X):
        # choose k random data points as initial centroids
        rng = np.random.RandomState()
        indices = rng.choice(len(X), size=self.k, replace=False)
        return X[indices].astype(float)

    def _compute_distances(self, X):
        if self.distance == 'euclidean':
            # Matrix expansion to avoid 3D array memory allocation (x-c)^2 = x^2 + c^2 - 2xc
            X_sq = np.sum(X**2, axis=1)[:, None]
            centers_sq = np.sum(self.centers**2, axis=1)[None, :]
            dot_product = np.dot(X, self.centers.T)
            
            sq_dists = X_sq + centers_sq - 2 * dot_product
            # np.maximum guarantees no negative numbers slip through due to floating-point precision
            return np.sqrt(np.maximum(sq_dists, 0.0))
            
        elif self.distance == 'manhattan':
            return np.abs(X[:, None, :] - self.centers[None, :, :]).sum(axis=2)

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.centers = self._seeds(X)
        self.sse_history_ = []

        for iteration in range(self.max_iters):
            # Step 1: Assign each point to the closest centroid
            dists = self._compute_distances(X)
            labels = np.argmin(dists, axis=1)

            # Step 2: Compute SSE directly to avoid extra function call overhead
            assigned_dists = dists[np.arange(len(X)), labels]
            sse = np.sum(assigned_dists ** 2)
            self.sse_history_.append(sse)

            # Step 3: Recompute centroids as mean of assigned points
            new_centers = np.zeros_like(self.centers)
            for j in range(self.k):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    new_centers[j] = cluster_points.mean(axis=0)
                else:
                    # Empty cluster — re-seed it randomly to avoid a dead centroid
                    new_centers[j] = X[np.random.randint(len(X))]

            # Step 4: Convergence check
            center_shift = np.linalg.norm(new_centers - self.centers)
            self.centers = new_centers

            if center_shift < self.tol:
                break

        # Final assignment with converged centers
        dists = self._compute_distances(X)
        self.labels_ = np.argmin(dists, axis=1)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        dists = self._compute_distances(X)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

class DBscan:  # 3-DBSCAN Clustering

    def __init__(self, eps=0.3, minpts=3, distance='euclidean'):
        # Validate the distance metric
        valid_distances = {"euclidean", "manhattan"}
        if distance not in valid_distances:
            raise ValueError(f"Distance must be one of: {valid_distances}")

        # Store model hyperparameters
        self.eps = eps
        self.minpts = minpts
        self.distance = distance

        # Cluster labels assigned after fitting
        self.labels_ = None

    def _compute_distance_matrix(self, X):
        # Compute the pairwise distance matrix using memory-efficient matrix expansion
        if self.distance == 'euclidean':
            X_sq = np.sum(X**2, axis=1)
            sq_dists = X_sq[:, None] + X_sq[None, :] - 2 * np.dot(X, X.T)
            return np.sqrt(np.maximum(sq_dists, 0.0))
            
        else:  # Manhattan distance
            return np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)

    def fit(self, X):
        X = np.array(X, dtype=float)
        n = len(X)

        # Compute pairwise distances between all samples
        dist_matrix = self._compute_distance_matrix(X)

        # Identify neighbors within the epsilon radius
        neighbor_mask = dist_matrix <= self.eps

        # Determine which points are core points
        is_core = neighbor_mask.sum(axis=1) >= self.minpts

        # Initialize all points as noise (-1)
        labels = np.full(n, -1)

        # Track visited points during cluster expansion
        visited = np.zeros(n, dtype=bool)

        # Cluster IDs start from 0
        cluster_id = 0

        # Visit each point in the dataset
        for i in range(n):
            # Skip points that are already processed or are not core points
            if visited[i] or not is_core[i]:
                continue

            # Start expanding a new cluster
            queue = [i]

            while queue:
                point = queue.pop()

                if visited[point]:
                    continue

                visited[point] = True
                labels[point] = cluster_id

                # Expand the cluster only from core points
                if is_core[point]:
                    neighbors = np.where(neighbor_mask[point])[0]
                    # Filter out already visited points to prevent queue explosion
                    unvisited_neighbors = neighbors[~visited[neighbors]]
                    queue.extend(unvisited_neighbors)

            # Move to the next cluster
            cluster_id += 1

        # Store the final cluster assignments
        self.labels_ = labels
        return self

    def fit_predict(self, X):
        # Fit the model and return the cluster labels
        self.fit(X)
        return self.labels_

class Perceptron:  # 4-Perceptron
# when activation='linear', loss='mse', it behaves as linear regression
# when activation='sigmoid', loss='log_loss', it behaves as logistic regression
    def __init__(self, lr=0.1, iterations=1000, activation='sigmoid', loss='mse'):
        # Validate the activation function
        valid_activations = {'sigmoid', 'step', 'relu', 'linear'}
        valid_losses = {'mse', 'log_loss'}

        if activation not in valid_activations:
            raise ValueError(f"Activation must be one of: {valid_activations}")
        if loss not in valid_losses:
            raise ValueError(f"Loss must be one of: {valid_losses}")

        # Store training hyperparameters
        self.lr = lr
        self.iterations = iterations
        self.activation = activation
        self.loss = loss

        # Model parameters learned during training
        self.weights = None
        self.bias = None
        self.loss_history_ = []

    # ---- Activation functions ----
    def _activate(self, z):
        # Apply the selected activation function
        if self.activation == 'sigmoid':
            z = np.clip(z, -500, 500)  # Prevent numerical overflow
            return 1.0 / (1.0 + np.exp(-z))

        elif self.activation == 'step':
            return np.where(z >= 0, 1, 0)

        elif self.activation == 'relu':
            return np.maximum(0, z)

        elif self.activation == 'linear':
            return z

    def _activation_derivative(self, z, a):
        # Compute the derivative of the selected activation function
        # 'a' is reused to avoid recomputing the activation output
        if self.activation == 'sigmoid':
            return a * (1 - a)

        elif self.activation == 'step':
            # Approximate the derivative since the step function is not differentiable
            return np.ones_like(z)

        elif self.activation == 'relu':
            return np.where(z > 0, 1, 0)

        elif self.activation == 'linear':
            return np.ones_like(z)

    # ---- Loss functions ----
    def _compute_loss(self, y, y_hat):
        # Compute the selected loss value
        if self.loss == 'mse':
            return np.mean((y_hat - y) ** 2)

        elif self.loss == 'log_loss':
            eps = 1e-9  # Prevent log(0)
            y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
            return -np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))

    def _loss_gradient(self, y, y_hat):
        # Compute the gradient of the loss with respect to the predictions
        if self.loss == 'mse':
            return 2 * (y_hat - y) / len(y)

        elif self.loss == 'log_loss':
            eps = 1e-9
            y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
            return -(y / y_hat_clipped - (1 - y) / (1 - y_hat_clipped)) / len(y)

    def fit(self, X, y):
        # Enforce strict NumPy array casting and dimension alignment
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).ravel()
        
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history_ = []

        # Train the model using gradient descent
        for _ in range(self.iterations):
            # Compute the linear output and activation
            z = np.dot(X, self.weights) + self.bias
            y_hat = self._activate(z)

            # Record the loss at the current iteration
            self.loss_history_.append(self._compute_loss(y, y_hat))

            # Mathematically stable fast-path for Logistic Regression
            if self.activation == 'sigmoid' and self.loss == 'log_loss':
                dz = (y_hat - y) / n_samples
            else:
                # Compute gradients using the chain rule for other combinations
                dloss_dyhat = self._loss_gradient(y, y_hat)
                dyhat_dz = self._activation_derivative(z, y_hat)
                dz = dloss_dyhat * dyhat_dz

            # Compute parameter gradients
            dw = np.dot(X.T, dz)
            db = np.sum(dz)

            # Update model parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X, t=0.5):
        # Compute the model output
        X = np.array(X, dtype=float)
        z = np.dot(X, self.weights) + self.bias
        y_raw = self._activate(z)

        # Vectorized probability conversion for classification
        if self.activation in ('sigmoid', 'step'):
            return (y_raw > t).astype(int)

        # Return raw predictions for other activation functions
        else:
            return y_raw

class NaiveBayes:  # 6-Naive Bayes Classifier

    def __init__(self):
        # Parameters learned during training
        self.classes_ = None
        self.mean_ = None      # Mean of each feature for every class
        self.var_ = None       # Variance of each feature for every class
        self.priors_ = None    # Prior probability of each class

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        # Identify the unique class labels
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Initialize parameter matrices
        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        # Estimate class-wise statistics
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]

            # Compute the mean and variance of each feature
            self.mean_[idx] = X_c.mean(axis=0)
            self.var_[idx] = X_c.var(axis=0) + 1e-9  # Prevent division by zero

            # Compute the prior probability of the class
            self.priors_[idx] = X_c.shape[0] / X.shape[0]

        return self

    def _gaussian_log_likelihood(self, X, class_idx):
        # Fetch precomputed mean and variance for the specific class
        mean = self.mean_[class_idx]
        var = self.var_[class_idx]

        # Compute the class-specific constant term once as a scalar
        const_term = -0.5 * np.sum(np.log(2 * np.pi * var))

        # Compute the sample-dependent exponential term and sum across features
        exp_term = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)

        return const_term + exp_term

    def _compute_log_posteriors(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Initialize the log posteriors matrix with shape (n_samples, n_classes)
        log_posteriors = np.zeros((n_samples, n_classes))

        # Compute the posterior for each class
        for idx in range(n_classes):
            log_prior = np.log(self.priors_[idx])
            log_likelihood = self._gaussian_log_likelihood(X, idx)

            # Apply Bayes' rule in log space
            log_posteriors[:, idx] = log_prior + log_likelihood

        return log_posteriors

    def predict(self, X):
        X = np.array(X, dtype=float)
        
        # Retrieve the log posterior probabilities
        log_posteriors = self._compute_log_posteriors(X)

        # Select the class with the highest posterior probability
        best_class_idx = np.argmax(log_posteriors, axis=1)
        return self.classes_[best_class_idx]

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        
        # Retrieve the log posterior probabilities
        log_posteriors = self._compute_log_posteriors(X)

        # Convert log probabilities into normalized probabilities
        # using the log-sum-exp trick for numerical stability
        log_posteriors -= log_posteriors.max(axis=1, keepdims=True)
        probs = np.exp(log_posteriors)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs  # Shape: (n_samples, n_classes)

class Node:  # Node used to represent each decision tree node
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:  # 7-Decision Tree
    def __init__(self, max_depth=5, min_samples_split=2, task='classification'):
        valid_tasks = {'classification', 'regression'}
        if task not in valid_tasks:
            raise ValueError(f"Task must be one of: {valid_tasks}")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.task = task
        self.root = None
        self.classes_ = None  # To store the original label mapping safely

    def _score(self, y):
        if self.task == 'classification':
            probs = np.bincount(y) / len(y)
            return 1.0 - np.sum(probs ** 2)
        else:
            return np.var(y) if len(y) > 0 else 0.0

    def _leaf_value(self, y):
        if self.task == 'classification':
            # y contains our internal 0-indexed mapping, we decode it to the original label
            majority_mapped_idx = np.bincount(y).argmax()
            return self.classes_[majority_mapped_idx]
        else:
            return np.mean(y)

    def _best_split(self, X, y):
        best_score, best_feature, best_thresh = float('inf'), None, None
        n_samples = len(y)

        # Pre-compute total sums/counts to enable O(1) subtraction for the right child
        if self.task == 'classification':
            n_classes = len(self.classes_)
            # Convert internal mapped y to one-hot for fast cumulative counting
            y_one_hot = np.eye(n_classes)[y]
            total_counts = y_one_hot.sum(axis=0)
        else:
            total_sum = np.sum(y)
            total_sq_sum = np.sum(y ** 2)

        for feature in range(X.shape[1]):
            # Optimization 1: Sort feature once per node -> O(n log n)
            sort_idx = np.argsort(X[:, feature])
            X_sorted = X[sort_idx, feature]
            y_sorted = y[sort_idx]

            # Optimization 2: Find unique split points
            # A valid split only occurs when the feature value actually changes
            valid_splits = X_sorted[:-1] != X_sorted[1:]
            
            if not np.any(valid_splits):
                continue

            # Optimization 3 & 4: Vectorized Incremental Impurity Updates -> O(n) sweep
            if self.task == 'classification':
                y_one_hot_sorted = y_one_hot[sort_idx]
                
                # Cumulative sum yields the exact class counts for ALL possible left splits instantly
                left_counts = np.cumsum(y_one_hot_sorted[:-1], axis=0)
                right_counts = total_counts - left_counts

                # Array of sizes for all possible left/right splits
                left_sizes = np.arange(1, n_samples)[:, None]
                right_sizes = n_samples - left_sizes

                # Vectorized Gini calculation for all N splits at once
                left_gini = 1.0 - np.sum((left_counts / left_sizes) ** 2, axis=1)
                right_gini = 1.0 - np.sum((right_counts / right_sizes) ** 2, axis=1)
                
                scores = (left_sizes[:, 0] * left_gini + right_sizes[:, 0] * right_gini) / n_samples
                
            else:  # Regression
                left_sums = np.cumsum(y_sorted[:-1])
                right_sums = total_sum - left_sums
                
                left_sq_sums = np.cumsum(y_sorted[:-1] ** 2)
                right_sq_sums = total_sq_sum - left_sq_sums
                
                left_sizes = np.arange(1, n_samples)
                right_sizes = n_samples - left_sizes
                
                # Vectorized Variance calculation: E[X^2] - (E[X])^2
                left_var = (left_sq_sums / left_sizes) - (left_sums / left_sizes) ** 2
                right_var = (right_sq_sums / right_sizes) - (right_sums / right_sizes) ** 2
                
                scores = (left_sizes * left_var + right_sizes * right_var) / n_samples

            # Isolate the impurity scores of strictly valid split points
            valid_scores = scores[valid_splits]
            valid_indices = np.where(valid_splits)[0]

            if len(valid_scores) > 0:
                min_score_idx = np.argmin(valid_scores)
                min_score = valid_scores[min_score_idx]
                
                if min_score < best_score:
                    best_score = min_score
                    best_feature = feature
                    # Optimization 2: Use exact midpoints between consecutive unique values
                    split_idx = valid_indices[min_score_idx]
                    best_thresh = (X_sorted[split_idx] + X_sorted[split_idx + 1]) / 2.0

        return best_feature, best_thresh

    def _build(self, X, y, depth=0):
        stop = depth >= self.max_depth or len(y) < self.min_samples_split or self._score(y) == 0
        if self.task == 'classification':
            stop = stop or len(np.unique(y)) == 1

        if stop:
            return Node(value=self._leaf_value(y))

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return Node(value=self._leaf_value(y))

        left_mask = X[:, feature] <= threshold
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)

        return Node(feature, threshold, left, right)

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        
        if self.task == 'classification':
            # SAFE LABEL MAPPING: np.unique maps ANY labels (strings, negative, sparse) to 0...C-1 internally.
            # This makes np.bincount and np.eye inside _best_split lightning fast and 100% bug-free.
            # The original labels are safely preserved in self.classes_ and injected directly into the Leaf nodes.
            self.classes_, y_mapped = np.unique(y, return_inverse=True)
            self.root = self._build(X, y_mapped)
        else:
            y = np.array(y, dtype=float)
            self.root = self._build(X, y)
            
        return self

    def _predict_one(self, x):
        # Iteratively traverse the tree to bypass Python recursion overhead
        node = self.root
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        return np.array([self._predict_one(x) for x in np.array(X, dtype=float)])    

class SVM:  # 8-Support Vector Machine

    def __init__(self, learning_rate=0.001, iterations=1000, C=1.0, kernel='linear', gamma=0.5):
        # Validate the kernel type
        valid_kernels = {'linear', 'rbf'}
        if kernel not in valid_kernels:
            raise ValueError(f"Kernel must be one of: {valid_kernels}")

        # Store model hyperparameters
        self.lr = learning_rate
        self.iterations = iterations
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

        # Model parameters learned during training
        self.weights = None
        self.bias = None
        self.X_train = None
        self.y_train = None
        self.alpha = None

    def _rbf(self, X1, X2):
        # Compute the RBF kernel matrix using vectorized pairwise squared distances
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        
        # Safe Optimization: Prevent floating-point inaccuracies from producing negative distances
        sq_dists = np.maximum(sq_dists, 0.0)
        return np.exp(-self.gamma * sq_dists)

    def fit(self, X, y):
        X = np.array(X, dtype=float)

        # Convert class labels to {-1, 1}
        y = np.where(np.array(y) <= 0, -1, 1)

        n_samples, n_features = X.shape

        if self.kernel == 'linear':
            # Initialize weights and bias
            self.weights = np.zeros(n_features)
            self.bias = 0.0

            # Train using gradient descent on the hinge loss objective (Sum of losses)
            for _ in range(self.iterations):
                scores = X @ self.weights + self.bias
                margin = y * scores

                # Identify samples violating the margin
                misclassified = (margin < 1).astype(float)

                # Compute gradients (Preserved original magnitude)
                dw = self.weights - self.C * (X.T @ (misclassified * y))
                db = -self.C * np.sum(misclassified * y)

                # Update model parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

        else:  # RBF kernel
            # Store the training data for kernel predictions
            self.X_train, self.y_train = X, y

            # Initialize dual parameters
            self.alpha = np.zeros(n_samples)
            self.bias = 0.0

            # Compute the kernel matrix once before training
            K = self._rbf(X, X)

            # Train using the dual representation (Preserved original dynamics)
            for _ in range(self.iterations):
                scores = (self.alpha * y) @ K + self.bias
                margin = y * scores

                # Identify samples violating the margin
                misclassified = (margin < 1).astype(float)

                # Update the dual coefficients
                self.alpha += self.lr * (misclassified - self.C * self.alpha)
                self.alpha = np.clip(self.alpha, 0, self.C)

                # Update the bias term
                self.bias += self.lr * np.sum(misclassified * y)

        return self

    def _decision_function(self, X):
        # Compute the decision score for each input sample
        X = np.array(X, dtype=float)

        if self.kernel == 'linear':
            return X @ self.weights + self.bias

        else:
            K = self._rbf(X, self.X_train)
            return K @ (self.alpha * self.y_train) + self.bias

    def predict(self, X):
        # Convert decision scores into binary class labels
        scores = self._decision_function(X)
        return np.where(scores >= 0, 1, 0)
    
class VotingEnsemble:
    """
    A custom ensemble meta-estimator that combines multiple base models.
    Supports both regression (weighted average) and classification (weighted hard voting).
    """
    def __init__(self, estimators, weights=None, task='classification'):
        if not estimators:
            raise ValueError("At least one estimator must be provided.")
        
        valid_tasks = {'classification', 'regression'}
        if task not in valid_tasks:
            raise ValueError(f"Task must be one of: {valid_tasks}")
            
        self.estimators = estimators
        self.task = task
        
        # Default to equal weights (1.0) if no custom weights are provided
        self.weights = weights if weights is not None else [1.0] * len(estimators)

    def fit(self, X, y):
        # Train every base estimator on the dataset
        for model in self.estimators:
            model.fit(X, y)
        return self

    def predict(self, X):
        # Collect predictions from all base estimators: shape (n_samples, n_estimators)
        predictions = np.column_stack([model.predict(X) for model in self.estimators])
        
        if self.task == 'regression':
            # Apply weighted average for regression
            return np.average(predictions, axis=1, weights=self.weights)
            
        elif self.task == 'classification':
            # Apply weighted hard voting for classification
            majority_votes = []
            for row in predictions:
                tally = {}
                for label, weight in zip(row, self.weights):
                    tally[label] = tally.get(label, 0.0) + weight
                    
                # Pick the label with the highest total weight
                best_label = max(tally, key=tally.get)
                majority_votes.append(best_label)
                
            return np.array(majority_votes)


#These were all the ml algorithms