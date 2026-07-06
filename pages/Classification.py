import streamlit as st
from models import KNN,Node,DecisionTree,SVM,NaiveBayes,Perceptron,VotingEnsemble
from datasets import get_classification_datasets
import matplotlib.pyplot as plt
import pandas as pd
models = {
    "KNN": KNN,
    "Decision Tree": DecisionTree,
    "Logistic Regression": Perceptron,
    "SVM": SVM,
    "Naive Bayes": NaiveBayes,
    "Perceptron": Perceptron,
}
st.set_page_config(
    page_title="Classification"
)

st.title("Classification")
mode = st.segmented_control(
    "Mode",
    ["Run Model", "Compare Models","Voting Ensemble"],
    default="Run Model"
)
with st.sidebar.expander("Dataset Settings", expanded=True):

    dataset_names = st.multiselect(
        "Datasets",
        options=["Moons", "Circles", "Blobs", "Classification"],
        default=["Moons"]
    )

    if not dataset_names:
        st.warning("Select at least one dataset.")
        st.stop()

    n_samples = st.slider(
        "Number of Samples",
        100,
        1000,
        300,
        100
    )

    resolution = st.slider(
        "Decision Boundary Resolution",
        20,
        150,
        60
    )

datasets = get_classification_datasets(n_samples)
selected_dataset = {
    name: datasets[name]
    for name in dataset_names
}
def get_model_ui(selected_model, key_prefix=""):
    """Renders hyperparameter UI and returns the instantiated model with unique Streamlit keys."""
    if selected_model == "KNN":
        k_knn = st.slider("K", 1, 20, 3, key=f"{key_prefix}_knn_k")
        distance_knn = st.selectbox("Distance", ["euclidean", "manhattan", "cosine"], key=f"{key_prefix}_knn_dist")
        return KNN(k=k_knn, distance=distance_knn, task='classification')

    elif selected_model == "Logistic Regression":
        lr_log = st.number_input("Learning Rate", value=0.01, step=0.001, format="%.4f", key=f"{key_prefix}_lr_log")
        iter_log = st.slider('Iterations', 0, 2000, 1000, key=f"{key_prefix}_iter_log")
        return Perceptron(lr=lr_log, iterations=iter_log,activation='sigmoid', loss='log_loss')

    elif selected_model == "Perceptron":
        lr_p = st.number_input("Learning Rate", value=0.01, step=0.001, format="%.4f", key=f"{key_prefix}_lr_p")
        iter_p = st.slider('Iterations', 0, 2000, 1000, key=f"{key_prefix}_iter_p")
        activation_p = st.selectbox("Activation", ['sigmoid', 'step'], key=f"{key_prefix}_act_p")
        loss_p = st.selectbox("Loss Function", ['mse', 'log_loss'], key=f"{key_prefix}_loss_p")
        
        # Educational info dynamically shown based on user selection (optional), 
        # or shown statically for all perceptron configs.
        if activation_p == 'sigmoid' and loss_p == 'log_loss':
            st.info("Note: With Activation as 'sigmoid' and Loss as 'log_loss', this Perceptron behaves exactly like Logistic Regression.")
            
        return Perceptron(lr=lr_p, activation=activation_p, iterations=iter_p, loss=loss_p)
    elif selected_model == "Naive Bayes":
        st.write("No hyperparameters to tune.")
        return NaiveBayes()

    elif selected_model == "Decision Tree":
        depth_dt = st.slider("Maximum Depth", 1, 20, 5, key=f"{key_prefix}_depth_dt")
        sample_split_dt = st.slider("Minimum Sample Split", 1, 20, 2, key=f"{key_prefix}_split_dt")
        return DecisionTree(max_depth=depth_dt, min_samples_split=sample_split_dt, task='classification')

    elif selected_model == "SVM":
        lr_svm = st.number_input("Learning Rate", value=0.01, step=0.001, format="%.4f", key=f"{key_prefix}_lr_svm")
        iter_svm = st.slider('Iterations', 0, 2000, 1000, key=f"{key_prefix}_iter_svm")
        kernel_svm = st.selectbox("Kernel", ['linear', 'rbf'], key=f"{key_prefix}_kernel_svm")
        C_svm = st.number_input("C", value=1.0, step=0.1, key=f"{key_prefix}_C_svm")
        if kernel_svm == "rbf":
            gamma_svm = st.number_input("Gamma", value=0.5, step=0.01, key=f"{key_prefix}_gamma_svm")
        else:
            gamma_svm = 0.5
        return SVM(learning_rate=lr_svm, iterations=iter_svm, C=C_svm, kernel=kernel_svm, gamma=gamma_svm)

def display_metrics(metrics):
    """Standardizes the display of evaluation metrics and the confusion matrix."""
    st.write(f"**Train Accuracy:** {metrics['train_accuracy']:.3f}")
    st.write(f"**Test Accuracy:** {metrics['test_accuracy']:.3f}")
    
    st.write("### Classification Report")
    # Requires 'import pandas as pd' at the top of your script
    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    st.dataframe(report_df, use_container_width=True)

    
    
# ==========================================
# MAIN LOGIC (Full Width)
# ==========================================

if mode == "Run Model":
    st.subheader("Model")
    selected_model = st.selectbox("Choose Model", list(models.keys()))
    
    with st.expander("Hyperparameters", expanded=True):
        # Instantiate using the helper function with a specific prefix
        model = get_model_ui(selected_model, key_prefix="run")
        
    if st.button("Run Model", use_container_width=True):
        from eval import plot_decision_boundary_fast,_fit_and_plot,model_eval
        with st.spinner("Running model..."):
            for fig, results in model_eval(model, datasets=selected_dataset, resolution=resolution):
                st.pyplot(fig)
                plt.close(fig)
                
                for dataset_name, metrics in results.items():
                    with st.expander(f" {dataset_name} Metrics", expanded=False):
                        display_metrics(metrics)
elif mode == "Voting Ensemble":
    st.subheader("Voting Ensemble")
    st.write("Assign weights to base models and compare the ensemble directly against them.")
    
    selected_estimators = st.multiselect(
        "Choose Base Models",
        options=list(models.keys()),
        default=["KNN", "Logistic Regression",'SVM']

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
        from models import VotingEnsemble
        from eval import model_eval # Use model_eval for Classification
        import pandas as pd
        
        # Instantiate the meta-estimator with custom weights
        ensemble_model = VotingEnsemble(estimators=models_to_ensemble, weights=ensemble_weights, task='classification')
        
        # Combine base models and the ensemble into a single comparison dictionary
        comparison_dict = {}
        for name, model_inst in zip(selected_estimators, models_to_ensemble):
            comparison_dict[name] = model_inst
        
        comparison_dict["Voting Ensemble"] = ensemble_model
        
        with st.spinner("Training models and generating comparison..."):
            # Pass the entire dictionary to your existing evaluation function
            for fig, results in model_eval(comparison_dict, datasets=selected_dataset):
                
                # 1. Render the plot for the current dataset
                st.pyplot(fig)
                plt.close(fig)
                
                # 2. Extract the current dataset name from the results dictionary keys
                current_ds = list(results.keys())[-1][0]
                
                # 3. Create a clean comparison DataFrame inside a collapsed expander
                with st.expander(f"📊 Metrics: {current_ds}", expanded=False):
                    
                    # Extract and flatten the nested metrics for the current dataset
                    flat_metrics = {}
                    for (ds_name, mod_name), metrics in results.items():
                        if ds_name == current_ds:
                            report = metrics["classification_report"]
                            
                            # Flattening the metrics so they look clean in a DataFrame table
                            flat_metrics[mod_name] = {
                                "Train Accuracy": metrics["train_accuracy"],
                                "Test Accuracy": metrics["test_accuracy"],
                                "Macro Precision": report["macro avg"]["precision"],
                                "Macro Recall": report["macro avg"]["recall"],
                                "Macro F1-Score": report["macro avg"]["f1-score"]
                            }
                    
                    # Create the DataFrame and Transpose (.T) so models are rows
                    df_metrics = pd.DataFrame(flat_metrics).T
                    
                    # Display the styled DataFrame
                    st.dataframe(
                        df_metrics.style.format("{:.4f}"),
                        use_container_width=True
                    )
else:
    st.subheader("Compare Models")
    
    # Enforce maximum of 4 models
    selected_models = st.multiselect(
        "Choose Models to Compare (Max 4)",
        options=list(models.keys()),
        default=["KNN", "Logistic Regression"],
        max_selections=4
    )
    
    models_to_run = {}
    
    if selected_models:
        with st.expander("Tune Hyperparameters", expanded=True):
            # Use tabs to cleanly separate the hyperparameter tuning for each model
            tabs = st.tabs(selected_models)
            for tab, model_name in zip(tabs, selected_models):
                with tab:
                    # Use a different prefix ("comp") to avoid widget ID collisions
                    models_to_run[model_name] = get_model_ui(model_name, key_prefix=f"comp_{model_name}")
                    
    if st.button("Compare Models", use_container_width=True, disabled=not selected_models):
        with st.spinner("Comparing models..."):
            from eval import plot_decision_boundary_fast, _fit_and_plot, model_eval
            import pandas as pd
            
            # Classification generator yields a figure and a results dict per dataset
            for fig, results in model_eval(models_to_run, datasets=selected_dataset, resolution=resolution):
                
                # 1. Render the plot for the current dataset
                st.pyplot(fig)
                plt.close(fig)
                
                # 2. Extract the current dataset name from the results dictionary keys
                current_ds = list(results.keys())[-1][0]
                
                # 3. Create a clean comparison DataFrame inside a collapsed expander
                with st.expander(f"📊 Metrics: {current_ds}", expanded=False):
                    
                    # Extract and flatten the nested metrics for the current dataset
                    flat_metrics = {}
                    for (ds_name, mod_name), metrics in results.items():
                        if ds_name == current_ds:
                            report = metrics["classification_report"]
                            
                            # Flattening the metrics so they look clean in a DataFrame table
                            flat_metrics[mod_name] = {
                                "Train Accuracy": metrics["train_accuracy"],
                                "Test Accuracy": metrics["test_accuracy"],
                                "Macro Precision": report["macro avg"]["precision"],
                                "Macro Recall": report["macro avg"]["recall"],
                                "Macro F1-Score": report["macro avg"]["f1-score"]
                            }
                    
                    # Create the DataFrame and Transpose (.T) so models are rows
                    df_metrics = pd.DataFrame(flat_metrics).T
                    
                    # Display the styled DataFrame
                    st.dataframe(
                        df_metrics.style.format("{:.4f}"),
                        use_container_width=True
                    )

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'>Designed & Developed by <b>Pranay Bothra</b> | Last Updated: July 2026</div>", unsafe_allow_html=True)