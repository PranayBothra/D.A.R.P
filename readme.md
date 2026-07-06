# D.A.R.P. | Dynamic Algorithm Rendering Platform
## A Deterministic,NumPy Engine for Machine Learning Diagnostics and Visualization

**D.A.R.P.** is an interactive machine learning environment designed for the strict deterministic evaluation and visual rendering of classical ML algorithms. Built to avoid the black-box nature of high-level APIs, the platform provides a transparent interface to directly observe algorithm mechanics, distance metrics, and gradient calculations. It allows users to dynamically adjust hyperparameters and instantly visualize how those changes impact model behavior, decision splits, and final evaluation metrics.

### Core Features
* **Pure NumPy Math Engine:** Over 10 classical machine learning algorithms (including SVM, K-Means, and Decision Trees) implemented entirely from scratch using pure Object-Oriented Python.
* **Dynamic Visual Rendering:** Real-time generation of complex 2D decision boundaries, regression curves, and density-based clustering assignments.
* **Interactive Hyperparameter Tuning:** Adjust parameters—such as learning rates, tree depths, or SVM kernels—on the fly and immediately observe the mathematical and visual impact.
* **Rigorous Metric Profiling:** Seamless evaluation and side-by-side comparison of models using dynamically generated classification reports, MSE, MAE, and R² scores.
* **Synthetic Data Pipelines:** Custom-generated datasets designed to stress-test specific algorithmic behaviors across classification, regression, and clustering tasks.

flowchart TD
    subgraph Data Pipeline
        DS[datasets.py]
    end

    subgraph Core Math Engine
        MOD[models.py]
        MOD_DESC(Pure NumPy OOP<br>Gradients, Splits, Metrics)
        MOD --- MOD_DESC
    end

    subgraph Evaluation Bridge
        EV[eval.py]
        EV_DESC(Matplotlib Rendering<br>Metric DataFrames)
        EV --- EV_DESC
    end

    subgraph Interactive Frontend
        UI[app.py & pages/]
    end

    DS -->|Synthetic Arrays & Labels| MOD
    MOD -->|Deterministic Predictions| EV
    EV -->|Visuals & Accuracy Reports| UI
    
    style MOD fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px
    style EV fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px