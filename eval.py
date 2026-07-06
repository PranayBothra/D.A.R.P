
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.lines import Line2D

DISTINCT_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#ff7f00',
    '#984ea3', '#ffff33', '#a65628', '#f781bf',
    '#999999', '#66c2a5', '#fc8d62', '#8da0cb',
    '#e78ac3', '#a6d854', '#ffd92f', '#e5c494'
]

# Plot the decision regions together with the training data
def plot_decision_boundary_fast(X, y, clf, ax, resolution=40):
    # Determine the plotting limits with a small padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create a grid of evenly spaced points across the feature space
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # SAFEGUARD: Chunk predictions to prevent Out-Of-Memory (OOM) RAM spikes
    chunk_size = 2000
    Z_chunks = []
    
    for i in range(0, len(grid_points), chunk_size):
        chunk = grid_points[i : i + chunk_size]
        # Predict on smaller batches to keep matrix allocations tiny
        Z_chunks.append(clf.predict(chunk))
        
    Z = np.concatenate(Z_chunks).reshape(xx.shape)

    # Build a color map based on the number of unique classes
    n_classes = len(np.unique(y))
    
    # Fallback applied in case n_classes exceeds DISTINCT_COLORS length
    # (Assuming DISTINCT_COLORS is defined globally in your script)
    cmap = ListedColormap((DISTINCT_COLORS * (n_classes // len(DISTINCT_COLORS) + 1))[:n_classes])

    # Display the decision regions
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap)

    # Overlay the training samples on top of the decision regions
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=25, linewidth=0.5)

    # Create custom legend entries for each class
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f"Class {cls}",
            markerfacecolor=cmap(i),
            markeredgecolor='k',
            markersize=8
        )
        for i, cls in enumerate(np.unique(y))
    ]

    # Add the legend to the plot
    ax.legend(
        handles=legend_elements,
        title="Classes",
        loc="upper right"
    )


# Train the classifier, evaluate its performance, and plot the decision boundary
def _fit_and_plot(ax, clf, X_train, y_train, X_test, y_test, title, resolution):
    # Fit the classifier on the training data
    clf.fit(X_train, y_train)

    # Generate predictions for both training and testing sets
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Calculate training and testing accuracy
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    # Flag the model if the train-test accuracy gap suggests overfitting
    gap = acc_train - acc_test
    overfit_flag = " overfit" if gap > 0.15 else ""

    # Visualize the learned decision boundary
    plot_decision_boundary_fast(
        X_train,
        y_train,
        clf,
        ax,
        resolution=resolution
    )

    # Display the model name along with its accuracy scores
    ax.set_title(
        f"{title}\n(train accuracy:{acc_train:.2f}, test accuracy:{acc_test:.2f}){overfit_flag}",
        fontsize=10
    )

    # Return the evaluation metrics
    return {
        "train_accuracy": acc_train,
        "test_accuracy": acc_test,
        "classification_report": classification_report(
            y_test,
            y_pred_test,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred_test
        )
    }


# Evaluate one or more models on one or more datasets
def model_eval(modele, datasets, resolution=40, max_cols=2):
    results = {}

    # Compare multiple models across each dataset
    if isinstance(modele, dict):
        models_dict = modele
        n_models = len(models_dict)
        n_cols = min(n_models, max_cols)
        n_rows = -(-n_models // n_cols)

        # Create a separate figure for each dataset
        for name, (X, y) in datasets.items():
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(7 * n_cols, 6 * n_rows),
                squeeze=False
            )
            axes = axes.ravel()

            # Train, evaluate, and visualize every model
            for i, (model_name, clf) in enumerate(models_dict.items()):
                metrics = _fit_and_plot(
                    axes[i],
                    clf,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    model_name,
                    resolution
                )

                results[(name, model_name)] = metrics

            # Hide any unused subplot axes
            for ax in axes[n_models:]:
                ax.axis('off')

            # Add the dataset name as the figure title
            fig.suptitle(name, fontsize=14, y=1.02)

            # SAFEGUARD: Thread-safe layout bound to the instance, not the global state
            fig.tight_layout()
            yield fig, results

    # Evaluate a single model across all datasets
    else:
        clf = modele
        n_datasets = len(datasets)
        n_cols = min(n_datasets, max_cols)
        n_rows = -(-n_datasets // n_cols)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(7 * n_cols, 6 * n_rows),
            squeeze=False
        )
        axes = axes.ravel()

        # Train, evaluate, and visualize the model on every dataset
        for i, (name, (X, y)) in enumerate(datasets.items()):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            metrics = _fit_and_plot(
                axes[i],
                clf,
                X_train,
                y_train,
                X_test,
                y_test,
                name,
                resolution
            )

            results[name] = metrics

        # Hide any unused subplot axes
        for ax in axes[n_datasets:]:
            ax.axis('off')

        # SAFEGUARD: Thread-safe layout
        fig.tight_layout()
        yield fig, results


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Evaluate one or more models on one or more datasets
def model_eval_regression(modele, datasets):
    # Accept either a single model or a dictionary of models
    if isinstance(modele, dict):
        models_dict = modele
    else:
        models_dict = {"Model": modele}

    results = {}
    n_datasets = len(datasets)

    # Use up to two columns while adapting automatically to the number of datasets
    n_cols = min(n_datasets, 2)

    # Compute the required number of subplot rows safely
    n_rows = -(-n_datasets // n_cols) if n_cols > 0 else 1

    # Create a figure with a size that scales according to the subplot layout
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        squeeze=False
    )
    axes = axes.ravel()

    # Color palette for plotting multiple regression models
    colors = plt.cm.tab10.colors

    # Evaluate every dataset
    for ax, (name, (X, y)) in zip(axes, datasets.items()):
        
        # SAFEGUARD: Prevent silent mathematical corruption. 
        # Reject datasets that cannot be inherently plotted on a 2D axis.
        if X.ndim > 1 and X.shape[1] > 1:
            raise ValueError(f"Dataset '{name}' has {X.shape[1]} features. model_eval_regression inherently supports only 1D feature spaces for 2D visualization.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Plot the training and testing samples safely using ravel()
        ax.scatter(
            X_train.ravel(),
            y_train,
            s=15,
            alpha=0.3,
            color="steelblue",
            label="Train data"
        )
        ax.scatter(
            X_test.ravel(),
            y_test,
            s=15,
            alpha=0.3,
            color="orange",
            label="Test data"
        )

        # Generate evenly spaced points for a smooth prediction curve
        X_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

        # Train, evaluate, and visualize each model
        for i, (model_name, clf) in enumerate(models_dict.items()):
            # Fit the model on the training data
            clf.fit(X_train, y_train)

            # Generate predictions for both training and testing sets
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)

            # Compute regression performance metrics
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)

            # Flag the model if the train-test R² gap suggests overfitting
            gap = r2_train - r2_test
            overfit_flag = " overfit" if gap > 0.15 else ""

            # Predict on a dense grid to draw a smooth regression curve
            y_line_pred = clf.predict(X_line)
            color = colors[i % len(colors)]
# VISUAL FIX: Make Ensemble lighter and transparent
            if model_name == "Voting Ensemble":
                line_color = "#E91E63"    
                line_style = "-"         # ":" creates a dotted line (since "o" throws an error)
                line_width = 1.5         # Slightly thicker so dots are visible
                z_order = 5              
            else:
                line_color = colors[i % len(colors)]
                line_style = "-"         # Solid line for base models
                line_width = 2
                z_order = 3              
                line_alpha = 1.0         # Base models stay 100% solid

            # Plot the fitted regression curve
            ax.plot(
                X_line,
                y_line_pred,
                linewidth=line_width,
                linestyle=line_style,
                color=line_color,
                zorder=z_order,
                label=f"{model_name} (Test R²:{r2_test:.2f}){overfit_flag}"
            )

            # # Plot the fitted regression curve
            # ax.plot(
            #     X_line,
            #     y_line_pred,
            #     linewidth=line_width,
            #     linestyle=line_style,
            #     color=line_color,
            #     zorder=z_order,
            #     label=f"{model_name} (Test R²:{r2_test:.2f}){overfit_flag}"
            # )
            # # Plot the fitted regression curve
            # ax.plot(
            #     X_line,
            #     y_line_pred,
            #     linewidth=2,
            #     color=color,
            #     label=f"{model_name} (Test R²:{r2_test:.2f}){overfit_flag}"
            # )

            # Store the evaluation metrics
            metrics_dict = {
                "train_r2": r2_train,
                "test_r2": r2_test,
                "train_mse": mse_train,
                "test_mse": mse_test,
                "train_mae": mae_train,
                "test_mae": mae_test
            }

            # Save results for either comparison mode or single-model mode
            if isinstance(modele, dict):
                results[(name, model_name)] = metrics_dict
            else:
                results[name] = metrics_dict

        # Set the dataset name as the subplot title
        ax.set_title(name)

        # Display the plot legend
        ax.legend(fontsize=8)

    # Hide any unused subplot axes
    for ax in axes[n_datasets:]:
        ax.axis('off')

    # SAFEGUARD: Thread-safe layout strictly bound to the local instance
    fig.tight_layout()

    # Return the figure and evaluation results
    yield fig, results



def _plot_single_cluster(ax, X, labels, title):
    # SAFEGUARD: Ensure dataset has at least 2 dimensions for a 2D scatter plot
    if X.ndim < 2 or X.shape[1] < 2:
        raise ValueError(f"Dataset requires at least 2 features for 2D clustering visualization, got {X.shape[1]}")

    # Separate noise (label -1) from valid clusters
    noise_mask = labels == -1
    valid_mask = ~noise_mask

    # VISUAL FIX: Plot noise points distinctly (muted grey 'x' markers)
    if noise_mask.any():
        ax.scatter(
            X[noise_mask, 0],
            X[noise_mask, 1],
            c='lightgray',
            marker='x',
            s=15,
            label='Noise'
        )

    # VISUAL & LOGIC FIX: Remap valid arbitrary cluster labels to consecutive integers (0, 1, 2...)
    # This forces the categorical colormap ('tab20') to use distinct, consistent colors
    if valid_mask.any():
        valid_labels = sorted(set(labels) - {-1})
        label_map = {lbl: i for i, lbl in enumerate(valid_labels)}
        mapped_labels = np.array([label_map[l] for l in labels[valid_mask]])

        ax.scatter(
            X[valid_mask, 0],
            X[valid_mask, 1],
            c=mapped_labels,
            cmap="tab20",
            s=20,
            edgecolor='k',
            linewidth=0.3
        )

    # Set the plot title and keep the aspect ratio equal
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


def model_eval_clustering(modele, datasets, max_cols=2):
    # Compare multiple clustering models on each dataset
    if isinstance(modele, dict):
        models_dict = modele
        n_models = len(models_dict)
        n_cols = min(n_models, max_cols)
        n_rows = -(-n_models // n_cols)

        # Create a separate figure for each dataset
        for name, (X, _) in datasets.items():
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(7 * n_cols, 6 * n_rows),
                squeeze=False
            )
            axes = axes.ravel()

            # Fit each model and visualize its clustering result
            for i, (model_name, clf) in enumerate(models_dict.items()):
                if hasattr(clf, "fit_predict"):
                    labels = clf.fit_predict(X)
                else:
                    clf.fit(X)
                    labels = clf.labels_

                _plot_single_cluster(axes[i], X, labels, model_name)

            # Hide any unused subplot axes
            for ax in axes[n_models:]:
                ax.axis('off')

            # Add the dataset name as the figure title
            fig.suptitle(name, fontsize=14, y=1.02)

            # SAFEGUARD: Thread-safe layout bound exclusively to this instance
            fig.tight_layout()
            yield fig

    # Evaluate a single clustering model across all datasets
    else:
        clf = modele
        n_datasets = len(datasets)
        n_cols = min(n_datasets, max_cols)
        n_rows = -(-n_datasets // n_cols)

        # Create the subplot layout
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(7 * n_cols, 6 * n_rows),
            squeeze=False
        )
        axes = axes.ravel()

        # Fit the model and visualize the clustering for every dataset
        for i, (name, (X, _)) in enumerate(datasets.items()):
            if hasattr(clf, "fit_predict"):
                labels = clf.fit_predict(X)
            else:
                clf.fit(X)
                labels = clf.labels_

            _plot_single_cluster(axes[i], X, labels, name)

        # Hide any unused subplot axes
        for ax in axes[n_datasets:]:
            ax.axis('off')

        # SAFEGUARD: Thread-safe layout
        fig.tight_layout()

        yield fig