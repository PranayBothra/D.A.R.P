import streamlit as st
import matplotlib.pyplot as plt
from models import Kmeans, DBscan
from datasets import get_clustering_datasets

# Define available models for Clustering
models = {
    "K-Means": Kmeans,
    "DBSCAN": DBscan
}

st.set_page_config(page_title="Clustering", layout="wide")

st.title("Clustering")
mode = st.segmented_control(
    "Mode",
    ["Run Model", "Compare Models"],
    default="Run Model"
)

with st.sidebar.expander("Dataset Settings", expanded=True):
    dataset_names = st.multiselect(
        "Datasets",
        options=["Moons", "Circles", "Blobs", "Smiley"],
        default=["Blobs"]
    )

    if not dataset_names:
        st.warning("Select at least one dataset.")
        st.stop()

    n_samples = st.slider("Number of Samples", 100, 1000, 500, 100)

# Fetch and filter datasets
datasets = get_clustering_datasets(n_samples)
selected_dataset = {name: datasets[name] for name in dataset_names}

def get_model_ui(selected_model, key_prefix=""):
    """Renders hyperparameter UI and returns the instantiated model."""
    if selected_model == "K-Means":
        k_clusters = st.slider("Clusters (K)", 1, 10, 3, key=f"{key_prefix}_kmeans_k")
        distance_kmeans = st.selectbox("Distance", ["euclidean", "manhattan"], key=f"{key_prefix}_kmeans_dist")
        max_iters = st.slider("Max Iterations", 100, 1000, 300, key=f"{key_prefix}_kmeans_iters")
        return Kmeans(k=k_clusters, distance=distance_kmeans, max_iters=max_iters)

    elif selected_model == "DBSCAN":
        eps = st.number_input("Epsilon (eps)", min_value=0.0,value=0.3, step=0.05, key=f"{key_prefix}_dbscan_eps")
        minpts = st.slider("Min Points", 1, 20, 3, key=f"{key_prefix}_dbscan_minpts")
        distance_dbscan = st.selectbox("Distance", ["euclidean", "manhattan"], key=f"{key_prefix}_dbscan_dist")
        return DBscan(eps=eps, minpts=minpts, distance=distance_dbscan)

if mode == "Run Model":
    st.subheader("Model")
    selected_model = st.selectbox("Choose Model", list(models.keys()))
    
    with st.expander("Hyperparameters", expanded=True):
        model = get_model_ui(selected_model, key_prefix="run")
        
    if st.button("Run Model", use_container_width=True):
        with st.spinner("Running model..."):
            from eval import model_eval_clustering
        # eval.py clustering evaluator returns either fig, or fig and an empty dict
            for item in model_eval_clustering(model, datasets=selected_dataset):
                fig = item[0] if isinstance(item, tuple) else item
                st.pyplot(fig)
                plt.close(fig)

else:
    st.subheader("Compare Models")
    selected_models = st.multiselect(
        "Choose Models to Compare",
        options=list(models.keys()),
        default=["K-Means", "DBSCAN"]
    )
    
    models_to_run = {}
    
    if selected_models:
        with st.expander("Tune Hyperparameters", expanded=True):
            tabs = st.tabs(selected_models)
            for tab, model_name in zip(tabs, selected_models):
                with tab:
                    models_to_run[model_name] = get_model_ui(model_name, key_prefix=f"comp_{model_name}")
                    
    if st.button("Compare Models", use_container_width=True, disabled=not selected_models):
        with st.spinner("Comparing models..."):
            from eval import model_eval_clustering
            for item in model_eval_clustering(models_to_run, datasets=selected_dataset):
                fig = item[0] if isinstance(item, tuple) else item
                st.pyplot(fig)
                plt.close(fig)


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'>Designed & Developed by <b>Pranay Bothra</b> | Last Updated: July 2026</div>", unsafe_allow_html=True)