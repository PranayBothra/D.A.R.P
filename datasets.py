from sklearn.datasets import (
    make_moons,
    make_circles,
    make_blobs,
    make_classification,
)
from sklearn.preprocessing import MinMaxScaler
import numpy as np
def get_classification_datasets(n_samples=300):
    return {
        "Moons": make_moons(
            n_samples=n_samples,
            noise=0.2
        ),
        "Circles": make_circles(
            n_samples=n_samples,
            noise=0.2,
            factor=0.4
        ),
        "Blobs": make_blobs(
            n_samples=n_samples,
            centers=2
        ),
        "Classification": make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1
        )
    }
def make_linear(n_samples=500, noise=1.2):
    rng = np.random.default_rng()
    X = np.linspace(-10, 10, n_samples).reshape(-1, 1)
    y = 2.5 * X.ravel() + 1 + rng.normal(0, noise, n_samples)
    return X, y

def make_polynomial(n_samples=500, noise=30):
    rng = np.random.default_rng()
    X = np.linspace(-10, 10, n_samples).reshape(-1, 1)
    y = 0.5 * X.ravel()**3 - 2 * X.ravel()**2 + X.ravel() + rng.normal(0, noise, n_samples)
    return X, y

def make_wavy(n_samples=500, noise=0.5):
    rng = np.random.default_rng()
    X = np.linspace(-10, 10, n_samples).reshape(-1, 1)
    y = np.sin(X.ravel()) * 3 + 0.3 * X.ravel() + rng.normal(0, noise, n_samples)
    return X, y

def make_quadratic(n_samples=500, noise=4):
    rng = np.random.default_rng()
    X = np.linspace(-10, 10, n_samples).reshape(-1, 1)
    y = 1.5 * X.ravel()**2 - 3 * X.ravel() + rng.normal(0, noise, n_samples)
    return X, y

def make_exponential(n_samples=500, noise=70):
    rng = np.random.default_rng()
    X = np.linspace(-10, 10, n_samples).reshape(-1, 1)
    y = np.exp(X.ravel() * 0.8) + rng.normal(0, noise, n_samples)
    return X, y

def get_regression_datasets(n_samples=500):
    return {
        "Linear": make_linear(n_samples=n_samples),
        "Quadratic": make_quadratic(n_samples=n_samples),
        "Polynomial (cubic)": make_polynomial(n_samples=n_samples),
        "Wavy (sine)": make_wavy(n_samples=n_samples),
        "Exponential": make_exponential(n_samples=n_samples),
    }


def make_smiley(n_samples=300, noise=0.03):
    rng = np.random.default_rng()
    # Allocate samples for the eyes and mouth
    n_eye = n_samples // 5
    n_mouth = n_samples - 2 * n_eye
    
    # Generate the left eye
    left_eye = rng.normal(loc=[-0.4, 0.3], scale=noise * 3, size=(n_eye, 2))
    
    # Generate the right eye
    right_eye = rng.normal(loc=[0.4, 0.3], scale=noise * 3, size=(n_eye, 2))
    
    # Generate the mouth as a noisy circular arc
    theta = rng.uniform(np.pi * 1.15, np.pi * 1.85, n_mouth)
    mouth_x = 0.7 * np.cos(theta)
    mouth_y = 0.7 * np.sin(theta) - 0.1
    mouth = np.column_stack([mouth_x, mouth_y]) + rng.normal(0, noise, (n_mouth, 2))
    
    # Combine all parts into a single dataset
    X = np.vstack([left_eye, right_eye, mouth])
    return X, None  # Labels are not required for clustering

def make_blobs_scaled(n_samples=500, centers=3):
    X, y = make_blobs(n_samples=n_samples, centers=centers)
    
    # Scale the data to approximately the same range as the other datasets
    X_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    return X_scaled, y

def get_clustering_datasets(n_samples=500):
    return {
        "Moons": make_moons(n_samples=n_samples, noise=0.05),
        "Circles": make_circles(n_samples=n_samples, noise=0.05, factor=0.4),
        "Blobs": make_blobs_scaled(n_samples=n_samples, centers=3),
        "Smiley": make_smiley(n_samples=n_samples),
    }