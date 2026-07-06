import streamlit as st

import streamlit as st

PROJECT_ACRONYM = "D.A.R.P."
PROJECT_FULL_NAME = "Dynamic Algorithm Rendering Platform"

st.set_page_config(
    page_title=f"{PROJECT_ACRONYM} | {PROJECT_FULL_NAME}",
    layout="wide"
)

# Introduction
with st.container(border=True):
    left, right = st.columns([3, 1])

    with left:
        st.title(f":blue[{PROJECT_ACRONYM}]")
        st.caption(PROJECT_FULL_NAME)

        st.markdown("""
Explore, visualize, and compare classical Machine Learning algorithms.

**Architectural Focus:** The core machine learning models in this platform are implemented entirely from scratch using pure Python and NumPy. By completely avoiding high-level machine learning APIs for model training, D.A.R.P. provides a transparent, deterministic interface to observe how distance metrics, gradient calculations, and decision splits render in real time.
""")

    with right:
        # st.metric("Algorithms", "10+")
        # st.metric("Categories", "3")
        st.markdown(":green[Algorithms]")
        st.caption("10+")
        st.markdown(":green[Categories]")
        st.caption("3")
        st.divider()
        st.markdown(":green[**Core Engine**]")
        st.caption("NumPy + OOP")

st.markdown("#### :blue[Platform Capabilities]")

cap1, cap2 = st.columns(2)

with cap1:
    with st.container(border=True):
        st.markdown("""
:orange[**Execution & Visualization**]

- Execute and compare deterministic models
- Tune hyperparameters with live rendering
- Visualize decision boundaries, clusters, and regression curves
""")

with cap2:
    with st.container(border=True):
        st.markdown("""
:orange[**Analysis & Evaluation**]

- Profile models using MSE, R², F1, and Macro Precision
- Explore synthetic data generation pipelines
- Inspect mathematical implementations from scratch
""")

st.divider()

# Features Section
st.subheader("Explore the Modules")
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("#### :blue[:material/category: Classification]")
        st.caption("Visualize decision boundaries, tune hyperparameters, and evaluate classification metrics across diverse datasets.")
        st.divider()
        st.markdown("""
        * **KNN**
        * **Logistic Regression**
        * **Perceptron**
        * **Naive Bayes**
        * **Decision Tree**
        * **SVM** (Linear & RBF)
        * **Voting Ensemble**
        """)
        st.page_link("pages/Classification.py", label="Go to Classification", icon=":material/arrow_forward:")

with col2:
    with st.container(border=True):
        st.markdown("#### :green[:material/show_chart: Regression]")
        st.caption("Fit continuous data, compare regression curves, and track MSE/MAE metrics seamlessly.")
        st.divider()
        st.markdown("""
        * **KNN Regressor**
        * **Linear Regression**
        * **Decision Tree Regressor**
        * **Voting Ensemble**
        """)
        st.page_link("pages/Regression.py", label="Go to Regression", icon=":material/arrow_forward:")

with col3:
    with st.container(border=True):
        st.markdown("#### :orange[:material/scatter_plot: Clustering]")
        st.caption("Discover hidden patterns in unsupervised data using distance and density-based clustering.")
        st.divider()
        st.markdown("""
        * **K-Means**
        * **DBSCAN**
        """)
        st.page_link("pages/Clustering.py", label="Go to Clustering", icon=":material/arrow_forward:")

# Tech Stack & Developer Info
st.subheader("Project Architecture")
st.write("The project architecture strictly separates the mathematical backend from the frontend UI. Here is how it operates:")

# Create a 2x2 grid for the architecture components
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True, height=220):
        st.markdown("#### :material/memory: `models.py` (Core Engine)")
        st.write("Pure Object-Oriented implementations of 8 base ML algorithms and a custom `VotingEnsemble`. The core math (gradients, distance metrics, Gini/Variance splits) is written entirely from scratch using **NumPy**.")

    with st.container(border=True, height=220):
        st.markdown("#### :material/query_stats: `eval.py` (Evaluation & Viz)")
        st.write("Acts as the bridge between models and the UI. It calculates performance metrics (via `sklearn.metrics`) and efficiently renders decision boundaries, regression curves, and scatter plots using strictly **Matplotlib**.")

with col2:
    with st.container(border=True, height=220):
        st.markdown("#### :material/dataset: `datasets.py` (Data Pipeline)")
        st.write("Generates synthetic datasets dynamically. Utilizes `sklearn.datasets` for standard shapes and custom **NumPy** logic for complex curves (sine waves, polynomials) and unique distributions like the Smiley dataset.")

    with st.container(border=True, height=220):
        st.markdown("#### :material/dashboard: `pages/` (Frontend UI)")
        st.write("**Streamlit**-powered interface managing hyperparameter inputs, state controls (Run vs. Compare vs. Ensemble), and dynamically rendering DataFrame comparisons without exposing backend complexity.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'>Designed & Developed by <b>Pranay Bothra</b> | Last Updated: July 2026</div>", unsafe_allow_html=True)