import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from models import KNN, DecisionTree,Perceptron,VotingEnsemble
from datasets import get_regression_datasets

# Define available models for Regression
models = {
    "KNN Regressor": KNN,
    "Decision Tree Regressor": DecisionTree,
    "Linear Regression": Perceptron
}

st.set_page_config(page_title="Regression", layout="wide")

st.title("Regression")
mode = st.segmented_control(
    "Mode",
    ["Run Model", "Compare Models","Voting Ensemble"],
    default="Run Model"
)

with st.sidebar.expander("Dataset Settings", expanded=True):
    dataset_names = st.multiselect(
        "Datasets",
        options=["Linear", "Quadratic", "Polynomial (cubic)", "Wavy (sine)", "Exponential"],
        default=["Linear"]
    )

    if not dataset_names:
        st.warning("Select at least one dataset.")
        st.stop()

    n_samples = st.slider("Number of Samples", 100, 1000, 500, 100)

# Fetch and filter datasets
datasets = get_regression_datasets(n_samples)
selected_dataset = {name: datasets[name] for name in dataset_names}

def get_model_ui(selected_model, key_prefix=""):
    """Renders hyperparameter UI and returns the instantiated model."""
    if selected_model == "KNN Regressor":
        k_knn = st.slider("K", 1, 20, 3, key=f"{key_prefix}_knn_k")
        distance_knn = st.selectbox("Distance", ["euclidean", "manhattan", "cosine"], key=f"{key_prefix}_knn_dist")
        return KNN(k=k_knn, distance=distance_knn, task='regression')

    elif selected_model == "Linear Regression":
        lr_lin = st.number_input("Learning Rate", value=0.01, step=0.001, format="%.4f", key=f"{key_prefix}_lr_lin")
        iter_lin = st.slider('Iterations', 0, 2000, 1000, key=f"{key_prefix}_iter_lin")
        return Perceptron(lr=lr_lin, iterations=iter_lin, activation='linear', loss='mse')

    elif selected_model == "Decision Tree Regressor":
        depth_dt = st.slider("Maximum Depth", 1, 20, 5, key=f"{key_prefix}_depth_dt")
        sample_split_dt = st.slider("Minimum Sample Split", 1, 20, 2, key=f"{key_prefix}_split_dt")
        return DecisionTree(max_depth=depth_dt, min_samples_split=sample_split_dt, task='regression')

def display_metrics(metrics):
    """Standardizes the display of evaluation metrics for regression."""
    st.write(f"**Train R²:** {metrics['train_r2']:.4f}")
    st.write(f"**Test R²:** {metrics['test_r2']:.4f}")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Train MSE:** {metrics['train_mse']:.4f}")
        st.write(f"**Train MAE:** {metrics['train_mae']:.4f}")
    with col2:
        st.write(f"**Test MSE:** {metrics['test_mse']:.4f}")
        st.write(f"**Test MAE:** {metrics['test_mae']:.4f}")


if mode == "Run Model":
    from eval import model_eval_regression
    st.subheader("Model")
    selected_model = st.selectbox("Choose Model", list(models.keys()))
    
    with st.expander("Hyperparameters", expanded=True):
        model = get_model_ui(selected_model, key_prefix="run")
        
    if st.button("Run Model", use_container_width=True):
        with st.spinner("Running model..."):
            from eval import model_eval_regression
            for fig, results in model_eval_regression(model, datasets=selected_dataset):
                st.pyplot(fig)
                plt.close(fig)
                
                for dataset_name, metrics in results.items():
                    with st.expander(f"{dataset_name} Metrics", expanded=False):
                        display_metrics(metrics)
elif mode == "Voting Ensemble":
    st.subheader("Voting Ensemble")
    st.write("Assign weights to base models and compare the ensemble directly against them.")
    
    selected_estimators = st.multiselect(
        "Choose Base Models",
        options=list(models.keys()),
        default=["KNN Regressor", "Decision Tree Regressor"], # Use Classifier names for Classification.py
    )
    
    models_to_ensemble = []
    ensemble_weights = []
    
    if selected_estimators:
        with st.expander("Tune Base Models & Assign Weights", expanded=True):
            tabs = st.tabs(selected_estimators)
            for tab, model_name in zip(tabs, selected_estimators):
                with tab:
                    # 1. Weight slider for each specific model
                    weight = st.slider(f"Voting Weight for {model_name}", min_value=1.0, max_value=10.0, value=1.0, step=0.5, key=f"weight_{model_name}")
                    ensemble_weights.append(weight)
                    
                    st.divider()
                    
                    # 2. Hyperparameter UI
                    model_instance = get_model_ui(model_name, key_prefix=f"ens_{model_name}")
                    models_to_ensemble.append(model_instance)
                    
    if st.button("Train & Compare Ensemble", use_container_width=True, disabled=not selected_estimators):

        # NOTE: Import VotingRegressor for Regression.py, and VotingClassifier for Classification.py
        from models import VotingEnsemble
        
        # Instantiate the meta-estimator with custom weights
        ensemble_model = VotingEnsemble(estimators=models_to_ensemble, weights=ensemble_weights,task='regression')
        
        #Combine base models and the ensemble into a single comparison dictionary
        comparison_dict = {}
        for name, model_inst in zip(selected_estimators, models_to_ensemble):
            comparison_dict[name] = model_inst
        
        comparison_dict["Voting Ensemble"] = ensemble_model
        with st.spinner("Training models and generating comparison..."):
            from eval import model_eval_regression
            import pandas as pd
            
        with st.spinner("Training models and generating comparison..."):
            from eval import model_eval_regression
            import pandas as pd
            
            # Regression yields ONCE with a combined figure and a full results dictionary
            for fig, results in model_eval_regression(comparison_dict, datasets=selected_dataset):
                
                # 1. Render the combined plot (contains subplots for all datasets)
                st.pyplot(fig)
                plt.close(fig)
                
                # 2. Find all unique dataset names present in the results
                unique_datasets = []
                for ds_name, mod_name in results.keys():
                    if ds_name not in unique_datasets:
                        unique_datasets.append(ds_name)
                        
            # 3. Render a separate DataFrame table for each dataset inside an expander
                for current_ds in unique_datasets:
                    
                    # VISUAL FIX: Put the entire table inside a collapsed expander
                    with st.expander(f"📊 Metrics: {current_ds}", expanded=False):
                        
                        # Extract metrics ONLY for the current dataset
                        metrics_for_current_ds = {}
                        for (ds_name, mod_name), metrics in results.items():
                            if ds_name == current_ds:
                                metrics_for_current_ds[mod_name] = metrics
                        
                        # Create and render the DataFrame
                        df_metrics = pd.DataFrame(metrics_for_current_ds).T
                        
                        # Reorder columns for logical reading
                        df_metrics = df_metrics[['train_r2', 'test_r2', 'train_mse', 'test_mse', 'train_mae', 'test_mae']]
                        
                        st.dataframe(
                            df_metrics.style.format("{:.4f}"),
                            use_container_width=True
                        )
else:
    st.subheader("Compare Models")
    selected_models = st.multiselect(
        "Choose Models to Compare",
        options=list(models.keys()),
        default=["KNN Regressor", "Linear Regression"]
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
            from eval import model_eval_regression
            import pandas as pd
            
            # Regression yields ONCE with a combined figure and a full results dictionary
            for fig, results in model_eval_regression(models_to_run, datasets=selected_dataset):
                
                # 1. Render the combined plot (contains subplots for all datasets)
                st.pyplot(fig)
                plt.close(fig)
                
                # 2. Find all unique dataset names present in the results
                unique_datasets = []
                for ds_name, mod_name in results.keys():
                    if ds_name not in unique_datasets:
                        unique_datasets.append(ds_name)
                        
                # 3. Render a separate DataFrame table for each dataset inside an expander
                for current_ds in unique_datasets:
                    
                    # VISUAL FIX: Put the entire table inside a collapsed expander
                    with st.expander(f"📊 Metrics: {current_ds}", expanded=False):
                        
                        # Extract metrics ONLY for the current dataset
                        metrics_for_current_ds = {}
                        for (ds_name, mod_name), metrics in results.items():
                            if ds_name == current_ds:
                                metrics_for_current_ds[mod_name] = metrics
                        
                        # Create and render the DataFrame
                        df_metrics = pd.DataFrame(metrics_for_current_ds).T
                        
                        # Reorder columns for logical reading
                        df_metrics = df_metrics[['train_r2', 'test_r2', 'train_mse', 'test_mse', 'train_mae', 'test_mae']]
                        
                        st.dataframe(
                            df_metrics.style.format("{:.4f}"),
                            use_container_width=True
                        )



st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'>Designed & Developed by <b>Pranay Bothra</b> | Last Updated: July 2026</div>", unsafe_allow_html=True)