# File: pycaret-ml-app/pages/evaluation.py
# Page for model evaluation functionality - Simplified and enhanced

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.model_training import save_model

def show():
    """Display the evaluation page"""
    st.header("📊 Model Evaluation")
    st.info("Step 5: Test your model on the test data you split earlier")

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
    target_column = processed_data['target_column']
    task_type = st.session_state.task_type
    model = st.session_state.trained_model
    best_model_name = st.session_state.get('best_model_name', 'Unknown Model')

    # Display best model info
    st.success(f"🏆 **Testing Model**: {best_model_name}")
    st.write(f"**Test Data Size**: {len(y_test)} samples")

    st.markdown("---")

    # Make predictions using PyCaret
    st.subheader("🎯 Testing on Your Test Data")

    with st.spinner("Generating predictions..."):
        try:
            # Use PyCaret's predict_model function
            if task_type == "classification":
                from pycaret.classification import predict_model
            else:
                from pycaret.regression import predict_model

            # Create test dataframe (target is already encoded during preprocessing)
            test_df = X_test.copy()
            test_df[target_column] = y_test.values

            # Make predictions
            predictions_df = predict_model(model, data=test_df)

            # Extract predictions
            y_pred = predictions_df['prediction_label'].values
            y_actual = y_test.values

        except Exception as e:
            st.error(f"❌ Error making predictions: {str(e)}")
            st.info("Trying alternative prediction method...")

            # Fallback: try direct prediction
            try:
                y_pred = model.predict(X_test)
                y_actual = y_test.values  # Already encoded during preprocessing
            except Exception as e2:
                st.error(f"❌ Alternative method also failed: {str(e2)}")
                import traceback
                with st.expander("🔍 View Error Details"):
                    st.code(traceback.format_exc())
                return

    # Calculate accuracy on test data
    if task_type == 'classification':
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_actual, y_pred)
        correct_predictions = int(accuracy * len(y_actual))
        total_samples = len(y_actual)

        # Big accuracy display
        st.markdown("---")
        st.subheader("📈 Test Results")

        # Show big accuracy number
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"<h1 style='text-align: center; color: green;'>{accuracy*100:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Accuracy on Test Data</h3>", unsafe_allow_html=True)

        st.write("")
        st.info(f"✅ **{correct_predictions}** correct predictions out of **{total_samples}** test samples")

    else:
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        total_samples = len(y_actual)

        # Big accuracy display for regression
        st.markdown("---")
        st.subheader("📈 Test Results")

        r2_percentage = r2 * 100

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h1 style='text-align: center; color: green;'>{r2_percentage:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Score</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>R² = {r2:.3f}</p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h1 style='text-align: center; color: blue;'>{mae:.2f}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Average Error</h3>", unsafe_allow_html=True)

        st.write("")
        st.info(f"📊 Tested on **{total_samples}** samples")

    st.markdown("---")

    # Visualization: Line chart comparing predictions vs actual
    st.subheader("📊 Predictions vs Actual (Test Data)")
    st.write("How well did the model predict your test data?")

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Index': range(len(y_actual)),
        'Actual': y_actual,
        'Predicted': y_pred
    })

    # Show first 20 samples
    display_df = comparison_df.head(20)

    if task_type == 'regression':
        # Line chart for regression
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(display_df['Index'], display_df['Actual'], marker='o', label='Actual Test Values', linewidth=2, markersize=8, color='blue')
        ax.plot(display_df['Index'], display_df['Predicted'], marker='s', label='Model Predictions', linewidth=2, markersize=8, color='orange')

        ax.set_xlabel('Test Sample Index')
        ax.set_ylabel('Value')
        ax.set_title('Test Data: Actual vs Predicted (First 20 Samples)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

    else:
        # Bar chart for classification
        display_df['Match'] = display_df['Actual'] == display_df['Predicted']

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['green' if match else 'red' for match in display_df['Match']]
        ax.scatter(display_df['Index'], display_df['Actual'], marker='o', s=100, label='Actual Test Values', alpha=0.6, color='blue')
        ax.scatter(display_df['Index'], display_df['Predicted'], marker='x', s=100, c=colors, label='Model Predictions (Green=Correct, Red=Wrong)', linewidths=2)

        ax.set_xlabel('Test Sample Index')
        ax.set_ylabel('Class')
        ax.set_title('Test Data: Actual vs Predicted (First 20 Samples)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()

    # Show detailed table
    with st.expander("📋 View Detailed Results (First 50 Test Samples)"):
        if task_type == 'classification':
            comparison_df['Match'] = (comparison_df['Actual'] == comparison_df['Predicted']).map({True: '✅ Correct', False: '❌ Wrong'})
        else:
            comparison_df['Error'] = abs(comparison_df['Actual'] - comparison_df['Predicted'])
            comparison_df['Error %'] = (comparison_df['Error'] / comparison_df['Actual'].abs() * 100).round(2)

        st.write(f"Showing predictions for the first 50 samples from your test data:")
        st.dataframe(comparison_df.head(50), use_container_width=True, hide_index=True)

    # Save Model Section
    st.markdown("---")
    st.subheader("💾 Save Model")
    st.write("Save your trained model for future use and deployment.")

    model_name = st.text_input(
        "Enter a name for your model:",
        value=f"{task_type}_best_model",
        help="This name will be used to save the model file"
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("💾 Save Model", use_container_width=True, type="primary"):
            with st.spinner("Saving model..."):
                success = save_model(model, model_name, task_type)
                if success:
                    st.session_state.saved_model_name = model_name
                    st.success(f"✅ Model saved successfully as '{model_name}'!")
                    st.info(f"📁 Location: models/{model_name}.pkl")

    with col2:
        if 'saved_model_name' in st.session_state:
            st.success("✅ Saved")

    # Summary
    st.markdown("---")
    st.subheader("✅ Summary")
    if task_type == 'classification':
        st.write(f"✓ **Model**: {best_model_name}")
        st.write(f"✓ **Test Accuracy**: {accuracy*100:.1f}%")
        st.write(f"✓ **Correct Predictions**: {correct_predictions}/{total_samples}")
    else:
        st.write(f"✓ **Model**: {best_model_name}")
        st.write(f"✓ **Test Score**: {r2_percentage:.1f}% (R² = {r2:.3f})")
        st.write(f"✓ **Average Error**: {mae:.2f}")
        st.write(f"✓ **Test Samples**: {total_samples}")

    # Next step button
    st.markdown("---")
    if st.button("➡️ Proceed to Prediction", use_container_width=True, type="primary"):
        st.session_state.page = "Prediction"
        st.rerun()
