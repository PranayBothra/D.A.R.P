<h1 align="center">D.A.R.P. | Dynamic Algorithm Rendering Platform</h1>
<h3 align="center">A Deterministic, NumPy Engine for Machine Learning Diagnostics and Visualization</h3>
<br>

**Overview:** D.A.R.P. is an interactive machine learning environment designed for the strict deterministic evaluation and visual rendering of classical ML algorithms. Built to bypass the hidden abstractions of high-level APIs, the platform provides a transparent interface to directly observe algorithm mechanics, distance metrics, and gradient calculations. 

Users can dynamically adjust hyperparameters and instantly visualize how those changes impact model behavior, decision splits, and final evaluation metrics.

### Core Capabilities

| Architecture & Features | Technical Implementation |
| :--- | :--- |
| **Pure NumPy Math Engine** | Over 10 classical machine learning algorithms (including SVM, K-Means, and Decision Trees) implemented entirely from scratch using pure Object-Oriented Python. |
| **Dynamic Visual Rendering** | Real-time generation of complex 2D decision boundaries, regression curves, and density-based clustering assignments. |
| **Interactive Parameter Tuning** | Adjust configurations—such as learning rates, tree depths, or SVM kernels—on the fly and immediately observe the mathematical and visual impact. |
| **Rigorous Metric Profiling** | Seamless evaluation and side-by-side comparison of models using dynamically generated classification reports, MSE, MAE, and R² scores. |
| **Synthetic Data Pipelines** | Custom-generated datasets designed to stress-test specific algorithmic behaviors across classification, regression, and clustering tasks. |

### System Architecture

```mermaid
flowchart TD
    A["datasets.py <br> Data Pipeline"] --->|Synthetic Data| B["models.py <br> Pure NumPy Engine"]
    B --->|Predictions| C["eval.py <br> Evaluation Bridge"]
    C --->|Visuals & Metrics| D["app.py & pages/ <br> Frontend"]
    
    style B fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px
    style C fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
