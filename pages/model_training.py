# File: pycaret-ml-app/pages/model_training.py
# Page for model training functionality - Simplified

import streamlit as st
import pandas as pd
import numpy as np
from src.model_training import initialize_setup, train_models, create_specific_model
from config.settings import CLASSIFICATION_METRICS, REGRESSION_METRICS

def show():
    """Display the model training page"""
    st.header("🤖 Model Training")
    st.info("Step 4: Train and compare machine learning models")

    # Check if processed data is available
    if 'processed_data' not in st.session_state:
        st.warning("⚠️ Please preprocess your data first.")
        if st.button("⬅️ Go to Preprocessing"):
            st.session_state.page = "Preprocessing"
            st.rerun()
        return

    # Get processed data
    processed_data = st.session_state.processed_data
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    target_column = processed_data['target_column']

    # Get or detect task type
    if 'task_type' not in st.session_state:
        # Auto-detect task type based on target variable
        y_unique_count = len(np.unique(y_train))
        st.session_state.task_type = "classification" if y_unique_count <= 10 else "regression"

    task_type = st.session_state.task_type

    # Display task information
    st.subheader("📊 Training Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", X_train.shape[0])
    with col2:
        st.metric("Features", X_train.shape[1])
    with col3:
        st.metric("Task Type", task_type.capitalize())
    with col4:
        st.metric("Target", target_column)

    # Show task-specific information
    if task_type == "classification":
        st.success("🎯 Classification Task: Predicting categories")
        st.write(f"**Evaluation Metrics**: {', '.join(CLASSIFICATION_METRICS)}")

        # Show class distribution
        with st.expander("📊 View Class Distribution"):
            class_counts = pd.Series(y_train).value_counts()
            class_df = pd.DataFrame({
                'Class': class_counts.index,
                'Count': class_counts.values,
                'Percentage': [f"{(v/len(y_train)*100):.1f}%" for v in class_counts.values]
            })
            st.dataframe(class_df, use_container_width=True, hide_index=True)

            # Warning for insufficient samples
            min_class_count = class_counts.min()
            if min_class_count < 2:
                st.error("⚠️ Error: At least one class has fewer than 2 samples. Model training will fail.")
                st.info("💡 Solution: Go back to Data Exploration and choose a different target column with more balanced classes.")
                return
    else:
        st.success("📈 Regression Task: Predicting numeric values")
        st.write(f"**Evaluation Metrics**: {', '.join(REGRESSION_METRICS)}")

        # Show target statistics
        with st.expander("📊 View Target Statistics"):
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    f"{y_train.mean():.2f}",
                    f"{y_train.median():.2f}",
                    f"{y_train.std():.2f}",
                    f"{y_train.min():.2f}",
                    f"{y_train.max():.2f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Initialize PyCaret setup
    st.markdown("---")
    st.subheader("🔧 Initialize ML Environment")

    st.write("PyCaret needs to be initialized before training models.")
    st.write("This step prepares your data for machine learning and may take a moment.")

    if st.button("🚀 Initialize Setup", use_container_width=True, type="primary"):
        with st.spinner("Initializing PyCaret... This may take 30-60 seconds"):
            # Create a combined dataframe for setup
            train_df = X_train.copy()
            train_df[target_column] = y_train.values

            # Show diagnostic info
            with st.expander("🔍 View Setup Diagnostics"):
                st.write(f"**Training DataFrame Shape**: {train_df.shape}")
                st.write(f"**Features (X)**: {X_train.shape[1]} columns")
                st.write(f"**Target Column**: {target_column}")
                st.write(f"**Target Sample Values**:")
                st.write(y_train.head(5))
                st.write(f"**Features Sample**:")
                st.dataframe(X_train.head(3), use_container_width=True)

            # Initialize setup
            setup = initialize_setup(train_df, target_column, task_type)

            if setup is not None:
                st.session_state.setup_initialized = True
                st.session_state.setup_initialized_for = task_type
                st.success("✅ Setup initialized successfully!")
                st.info("You can now train and compare models below.")
                st.rerun()

    # Show model training options only after setup is initialized
    if 'setup_initialized' in st.session_state and st.session_state.setup_initialized:
        st.markdown("---")
        st.subheader("🎯 Model Selection & Comparison")
        st.write("Select the models you want to train and compare. The system will rank them by performance and save the best one.")
        st.write("Choose specific models you want to compare, then the system will rank them by performance.")

        # Model selection with checkboxes
        if task_type == "classification":
            available_models = {
                'Logistic Regression': 'lr',
                'Decision Tree': 'dt',
                'Random Forest': 'rf',
                'K-Nearest Neighbors': 'knn',
                'Naive Bayes': 'nb',
                'Support Vector Machine': 'svm',
                'Gradient Boosting (XGBoost)': 'xgboost',
                'LightGBM': 'lightgbm'
            }
        else:
            available_models = {
                'Linear Regression': 'lr',
                'Lasso Regression': 'lasso',
                'Ridge Regression': 'ridge',
                'Elastic Net': 'en',
                'Decision Tree': 'dt',
                'Random Forest': 'rf',
                'K-Nearest Neighbors': 'knn',
                'Gradient Boosting (XGBoost)': 'xgboost',
                'LightGBM': 'lightgbm'
            }

        st.write("Select the models you want to compare:")

        # Create columns for better layout
        col1, col2, col3 = st.columns(3)

        selected_models = []
        model_names = list(available_models.keys())

        # Distribute models across columns
        for i, model_name in enumerate(model_names):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.checkbox(model_name, key=f"model_select_{model_name}"):
                    selected_models.append((model_name, available_models[model_name]))

        # Show selected count
        if selected_models:
            st.info(f"✅ {len(selected_models)} model(s) selected: {', '.join([m[0] for m in selected_models])}")
        else:
            st.warning("⚠️ Please select at least 2 models to compare")

        # Compare selected models button
        if len(selected_models) >= 2:
            if st.button(f"🏆 Compare Selected {len(selected_models)} Models", use_container_width=True, type="primary"):
                with st.spinner(f"Training and comparing {len(selected_models)} selected models..."):
                    try:
                        # Use PyCaret's compare_models with include parameter
                        from pycaret.classification import compare_models as clf_compare, pull as clf_pull, predict_model as clf_predict
                        from pycaret.regression import compare_models as reg_compare, pull as reg_pull, predict_model as reg_predict
                        from sklearn.metrics import accuracy_score, r2_score
                        import matplotlib.pyplot as plt

                        # Get only the model IDs
                        selected_model_ids = [m[1] for m in selected_models]

                        # Compare models using PyCaret's built-in function
                        st.info(f"Training {len(selected_models)} models...")

                        if task_type == "classification":
                            # Compare classification models
                            best_model = clf_compare(include=selected_model_ids, n_select=1, verbose=False)
                            comparison_df = clf_pull()
                        else:
                            # Compare regression models
                            best_model = reg_compare(include=selected_model_ids, n_select=1, verbose=False)
                            comparison_df = reg_pull()

                        # Display comparison results
                        if comparison_df is not None and not comparison_df.empty:
                            st.success("✅ Model comparison complete!")

                            # Sort by main metric to get correct order
                            if task_type == "classification":
                                sort_metric = 'Accuracy'
                            else:
                                sort_metric = 'R2'

                            # Sort comparison dataframe by the metric (descending)
                            if sort_metric in comparison_df.columns:
                                comparison_df = comparison_df.sort_values(by=sort_metric, ascending=False)

                            # Get test data to show actual predictions
                            X_test = processed_data['X_test']
                            y_test = processed_data['y_test']

                            # Create test dataframe
                            test_df = X_test.copy()
                            test_df[target_column] = y_test.values

                            st.markdown("---")
                            st.subheader("📊 Model Performance Comparison")
                            st.write("Testing each model on your test data to find the best one:")

                            # Store all models and their performance
                            model_performances = []

                            # Train each model and get predictions
                            for model_name, model_id in selected_models:
                                st.write(f"**{model_name}**")

                                try:
                                    # Create model
                                    if task_type == "classification":
                                        from pycaret.classification import create_model as clf_create
                                        model = clf_create(model_id, verbose=False)
                                        predictions_df = clf_predict(model, data=test_df)
                                        y_pred = predictions_df['prediction_label'].values

                                        # y_test is already encoded during preprocessing
                                        y_actual = y_test.values

                                        # Calculate accuracy
                                        accuracy = accuracy_score(y_actual, y_pred)
                                        model_performances.append((model_name, accuracy, model))

                                        # Show accuracy as percentage
                                        correct = int(accuracy * len(y_actual))
                                        total = len(y_actual)

                                        col1, col2 = st.columns([1, 3])
                                        with col1:
                                            st.metric("Test Accuracy", f"{accuracy*100:.1f}%")
                                        with col2:
                                            st.write(f"✅ {correct} correct / ❌ {total - correct} wrong (out of {total} samples)")

                                    else:
                                        from pycaret.regression import create_model as reg_create
                                        model = reg_create(model_id, verbose=False)
                                        predictions_df = reg_predict(model, data=test_df)
                                        y_pred = predictions_df['prediction_label'].values
                                        y_actual = y_test.values

                                        # Calculate R2
                                        r2 = r2_score(y_actual, y_pred)
                                        model_performances.append((model_name, r2, model))

                                        # Show R2 as percentage
                                        r2_percentage = r2 * 100
                                        total = len(y_actual)

                                        col1, col2 = st.columns([1, 3])
                                        with col1:
                                            st.metric("Test Score", f"{r2_percentage:.1f}%")
                                        with col2:
                                            st.write(f"📊 R² = {r2:.3f} (tested on {total} samples)")

                                    # Show small chart for this model
                                    fig, ax = plt.subplots(figsize=(10, 4))

                                    # Show first 15 samples
                                    display_samples = 15
                                    indices = range(min(display_samples, len(y_actual)))

                                    if task_type == "regression":
                                        ax.plot(indices, y_actual[:display_samples], marker='o', label='Actual Values', linewidth=2, markersize=6, color='blue')
                                        ax.plot(indices, y_pred[:display_samples], marker='s', label='Predicted by Model', linewidth=2, markersize=6, color='orange')
                                        ax.set_ylabel('Value')
                                        ax.set_title(f'How well does {model_name} predict? (First 15 test samples)')
                                    else:
                                        matches = y_actual[:display_samples] == y_pred[:display_samples]
                                        colors = ['green' if m else 'red' for m in matches]
                                        ax.scatter(indices, y_actual[:display_samples], marker='o', s=80, label='Actual Values', alpha=0.6, color='blue')
                                        ax.scatter(indices, y_pred[:display_samples], marker='x', s=80, c=colors, label='Predictions (Green=✓, Red=✗)', linewidths=2)
                                        ax.set_ylabel('Class')
                                        ax.set_title(f'How well does {model_name} predict? (First 15 test samples)')

                                    ax.set_xlabel('Test Sample')
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)

                                    st.pyplot(fig)
                                    plt.close()

                                    st.markdown("---")

                                except Exception as e:
                                    st.error(f"Error with {model_name}: {str(e)}")

                            # Identify best model
                            if model_performances:
                                # Sort by performance (higher is better)
                                model_performances.sort(key=lambda x: x[1], reverse=True)
                                best_model_name = model_performances[0][0]
                                best_model = model_performances[0][2]
                                best_performance = model_performances[0][1]

                                st.markdown("---")
                                st.subheader("📋 Comparison Table")
                                st.write("**Which model is the best? Compare the numbers:**")

                                # Create simple comparison table
                                if task_type == "classification":
                                    comparison_table = pd.DataFrame({
                                        'Model': [m[0] for m in model_performances],
                                        'Accuracy': [f"{m[1]*100:.1f}%" for m in model_performances]
                                    })
                                else:
                                    comparison_table = pd.DataFrame({
                                        'Model': [m[0] for m in model_performances],
                                        'Score': [f"{m[1]*100:.1f}%" for m in model_performances],
                                        'R²': [f"{m[1]:.3f}" for m in model_performances]
                                    })

                                # Add ranking
                                comparison_table.insert(0, 'Rank', [f"#{i+1}" for i in range(len(model_performances))])

                                # Highlight best model row
                                st.dataframe(
                                    comparison_table,
                                    use_container_width=True,
                                    hide_index=True
                                )

                                st.markdown("---")
                                st.success(f"🏆 **Winner: {best_model_name}**")
                                if task_type == "classification":
                                    st.write(f"✅ **Best Test Accuracy**: {best_performance*100:.1f}%")
                                else:
                                    st.write(f"✅ **Best Test Score**: {best_performance*100:.1f}% (R² = {best_performance:.3f})")

                                # Store results in session state
                                st.session_state.trained_model = best_model
                                st.session_state.best_model_name = best_model_name
                                st.session_state.comparison_results = comparison_df

                                st.info("➡️ Go to the Evaluation page to see detailed test results for the best model.")
                        else:
                            st.error("❌ No comparison results returned. Please try again.")

                    except Exception as e:
                        st.error(f"❌ Error during model comparison: {str(e)}")
                        import traceback
                        with st.expander("🔍 View Error Details"):
                            st.code(traceback.format_exc())
        elif len(selected_models) == 1:
            st.warning("⚠️ Please select at least 2 models to compare.")

    # Next step button
    if 'trained_model' in st.session_state:
        st.markdown("---")
        st.success("✅ Model training complete!")
        if st.button("➡️ Proceed to Evaluation", use_container_width=True, type="primary"):
            st.session_state.page = "Evaluation"
            st.rerun()
