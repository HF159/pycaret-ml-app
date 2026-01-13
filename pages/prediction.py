# File: pycaret-ml-app/pages/prediction.py
# Page for prediction functionality - Simplified

import streamlit as st
import pandas as pd
from src.prediction import export_predictions

def show():
    """Display the prediction page"""
    st.header("🔮 Make Predictions")
    st.info("Step 6: Use your trained model to make predictions")

    # Check if trained model is available
    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train a model first.")
        if st.button("⬅️ Go to Model Training"):
            st.session_state.page = "Model Training"
            st.rerun()
        return

    # Get data from session state
    processed_data = st.session_state.processed_data
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    task_type = st.session_state.task_type
    model = st.session_state.trained_model

    # Prediction on Test Data
    st.subheader("📊 Predictions on Test Data")
    st.write("Generate predictions on the test set that was reserved during preprocessing.")

    if st.button("🔮 Generate Test Predictions", use_container_width=True, type="primary"):
        with st.spinner("Generating predictions..."):
            try:
                # Create a dataframe with predictions
                df_pred = X_test.copy()
                df_pred['Actual'] = y_test.values if hasattr(y_test, 'values') else y_test

                # Make predictions
                y_pred = model.predict(X_test)
                df_pred['Predicted'] = y_pred

                # For classification, add probabilities if available
                if task_type == 'classification':
                    try:
                        y_proba = model.predict_proba(X_test)
                        classes = model.classes_

                        for i, cls in enumerate(classes):
                            df_pred[f'Probability_{cls}'] = y_proba[:, i]
                    except:
                        pass

                # Calculate accuracy for classification or error for regression
                if task_type == 'classification':
                    accuracy = (df_pred['Actual'] == df_pred['Predicted']).mean()
                    st.success(f"✅ Predictions generated! Test Accuracy: {accuracy:.2%}")
                else:
                    from sklearn.metrics import mean_absolute_error, r2_score
                    mae = mean_absolute_error(df_pred['Actual'], df_pred['Predicted'])
                    r2 = r2_score(df_pred['Actual'], df_pred['Predicted'])
                    st.success(f"✅ Predictions generated! R² Score: {r2:.4f}, MAE: {mae:.4f}")

                # Display sample predictions
                st.subheader("👀 Sample Predictions")
                st.dataframe(df_pred.head(10), use_container_width=True)

                # Summary statistics
                with st.expander("📊 View Prediction Summary"):
                    st.write(f"**Total Predictions**: {len(df_pred)}")

                    if task_type == 'classification':
                        st.write("**Prediction Distribution:**")
                        pred_counts = df_pred['Predicted'].value_counts()
                        pred_df = pd.DataFrame({
                            'Class': pred_counts.index,
                            'Count': pred_counts.values,
                            'Percentage': [f"{(v/len(df_pred)*100):.1f}%" for v in pred_counts.values]
                        })
                        st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    else:
                        st.write("**Prediction Statistics:**")
                        stats_df = pd.DataFrame({
                            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                            'Value': [
                                f"{df_pred['Predicted'].mean():.2f}",
                                f"{df_pred['Predicted'].median():.2f}",
                                f"{df_pred['Predicted'].std():.2f}",
                                f"{df_pred['Predicted'].min():.2f}",
                                f"{df_pred['Predicted'].max():.2f}"
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)

                # Download button
                csv = export_predictions(df_pred, "test_predictions.csv")
                st.download_button(
                    label="⬇️ Download All Test Predictions (CSV)",
                    data=csv,
                    file_name="test_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                import traceback
                with st.expander("🔍 View Error Details"):
                    st.code(traceback.format_exc())

    # Prediction on New Data
    st.markdown("---")
    st.subheader("📁 Predictions on New Data")
    st.write("Upload a new dataset to generate predictions.")
    st.info("ℹ️ **Important**: The new data must have the same columns as the training data (excluding the target column).")

    # Show required columns
    with st.expander("📋 View Required Columns"):
        required_cols = X_test.columns.tolist()
        st.write("Your new data file must contain these columns:")
        for col in required_cols:
            st.write(f"• {col}")

    # File upload for new data
    new_data_file = st.file_uploader(
        "Upload new data file (CSV, Excel, or JSON):",
        type=["csv", "xlsx", "json"],
        help="Upload a file with the same structure as your training data"
    )

    if new_data_file is not None:
        try:
            # Load the file
            if new_data_file.name.endswith('.csv'):
                new_data = pd.read_csv(new_data_file)
            elif new_data_file.name.endswith('.xlsx'):
                new_data = pd.read_excel(new_data_file, engine='openpyxl')
            else:  # JSON
                new_data = pd.read_json(new_data_file)

            st.success(f"✅ File uploaded: {new_data_file.name}")

            # Display preview
            st.write("**Data Preview:**")
            st.dataframe(new_data.head(), use_container_width=True)

            # Check for required columns
            required_cols = X_test.columns.tolist()
            missing_cols = [col for col in required_cols if col not in new_data.columns]
            extra_cols = [col for col in new_data.columns if col not in required_cols]

            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your file contains all the required columns listed above.")
            else:
                # Show column status
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(new_data))
                with col2:
                    st.metric("Required Columns", len(required_cols))
                with col3:
                    if extra_cols:
                        st.metric("Extra Columns", len(extra_cols))
                        st.caption("(Will be ignored)")

                # Make predictions button
                if st.button("🔮 Generate Predictions", use_container_width=True, type="primary"):
                    with st.spinner("Making predictions..."):
                        try:
                            # Select only required columns in the correct order
                            new_data_processed = new_data[required_cols]

                            # Make predictions
                            y_pred = model.predict(new_data_processed)

                            # Create results dataframe
                            results = new_data.copy()
                            results['Predicted'] = y_pred

                            # For classification, add probabilities
                            if task_type == 'classification':
                                try:
                                    y_proba = model.predict_proba(new_data_processed)
                                    classes = model.classes_

                                    for i, cls in enumerate(classes):
                                        results[f'Probability_{cls}'] = y_proba[:, i]
                                except:
                                    pass

                            st.success(f"✅ Generated {len(results)} predictions successfully!")

                            # Display results
                            st.subheader("📊 Prediction Results")
                            st.dataframe(results.head(10), use_container_width=True)

                            # Summary
                            with st.expander("📊 View Summary"):
                                if task_type == 'classification':
                                    pred_counts = results['Predicted'].value_counts()
                                    pred_df = pd.DataFrame({
                                        'Class': pred_counts.index,
                                        'Count': pred_counts.values,
                                        'Percentage': [f"{(v/len(results)*100):.1f}%" for v in pred_counts.values]
                                    })
                                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                                else:
                                    stats_df = pd.DataFrame({
                                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                                        'Value': [
                                            f"{results['Predicted'].mean():.2f}",
                                            f"{results['Predicted'].median():.2f}",
                                            f"{results['Predicted'].std():.2f}",
                                            f"{results['Predicted'].min():.2f}",
                                            f"{results['Predicted'].max():.2f}"
                                        ]
                                    })
                                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

                            # Download button
                            csv = export_predictions(results, "new_data_predictions.csv")
                            st.download_button(
                                label="⬇️ Download All Predictions (CSV)",
                                data=csv,
                                file_name="new_data_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        except Exception as e:
                            st.error(f"❌ Error making predictions: {str(e)}")
                            import traceback
                            with st.expander("🔍 View Error Details"):
                                st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    # Return to home button
    st.markdown("---")
    if st.button("🏠 Start New Project", use_container_width=True):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.page = "Data Upload"
        st.rerun()
