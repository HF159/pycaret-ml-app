# File: pycaret-ml-app/pages/prediction.py
# Page for prediction functionality

import streamlit as st
import pandas as pd
import numpy as np
from src.prediction import predict_model, load_model, process_input_data, interpret_predictions, generate_input_form, export_predictions

def show():
    """Display the prediction page"""
    st.header("Make Predictions")
    
    # Check if trained model is available
    if 'trained_model' not in st.session_state:
        st.warning("Please train a model first.")
        if st.button("Go to Model Training"):
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
    preprocessing_metadata = processed_data['preprocessing_metadata']
    
    # Prediction on test data
    st.subheader("Predictions on Test Data")
    
    if st.button("Generate Predictions on Test Data"):
        with st.spinner("Generating predictions..."):
            # Create a dataframe with true and predicted values
            df_pred = X_test.copy()
            df_pred['Actual'] = y_test
            
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
            
            # Display predictions
            st.dataframe(df_pred.head(10))
            
            # Download button for full predictions
            csv = export_predictions(df_pred, "test_predictions.csv")
            st.download_button(
                label="Download All Test Predictions",
                data=csv,
                file_name="test_predictions.csv",
                mime="text/csv"
            )
    
    # Prediction on new data
    st.subheader("Predictions on New Data")
    
    prediction_method = st.radio(
        "Select prediction method",
        ["Upload File", "Manual Input"]
    )
    
    if prediction_method == "Upload File":
        # File upload for new data
        st.write("Upload new data for prediction")
        new_data_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])
        
        if new_data_file is not None:
            try:
                # Load file
                if new_data_file.name.endswith('.csv'):
                    new_data = pd.read_csv(new_data_file)
                elif new_data_file.name.endswith('.xlsx'):
                    new_data = pd.read_excel(new_data_file, engine='openpyxl')
                else:  # JSON
                    new_data = pd.read_json(new_data_file)
                
                # Display data preview
                st.write("Data Preview:")
                st.dataframe(new_data.head(5))
                
                # Check if required columns are present
                required_cols = X_test.columns
                missing_cols = [col for col in required_cols if col not in new_data.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    # Process the data
                    processed_new_data = process_input_data(new_data, preprocessing_metadata)
                    
                    # Make predictions button
                    if st.button("Make Predictions on Uploaded Data"):
                        with st.spinner("Making predictions..."):
                            # Make predictions
                            predictions = predict_model(model, processed_new_data, task_type)
                            
                            # Display predictions
                            if predictions is not None:
                                interpreted_predictions = interpret_predictions(predictions, task_type)
                                st.dataframe(interpreted_predictions.head(10))
                                
                                # Download button for full predictions
                                csv = export_predictions(interpreted_predictions, "new_data_predictions.csv")
                                st.download_button(
                                    label="Download All Predictions",
                                    data=csv,
                                    file_name="new_data_predictions.csv",
                                    mime="text/csv"
                                )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:  # Manual Input
        st.write("Enter values manually for prediction")
        
        # Generate input form based on features
        original_df = st.session_state.data
        input_df = generate_input_form(original_df, target_column)
        
        if input_df is not None and st.button("Make Prediction on Manual Input"):
            with st.spinner("Making prediction..."):
                try:
                    # Process the input data
                    processed_input = process_input_data(input_df, preprocessing_metadata)
                    
                    # Make prediction
                    prediction = model.predict(processed_input)[0]
                    
                    # Display result
                    st.success("Prediction Results:")
                    
                    if task_type == 'classification':
                        st.write(f"Predicted Class: {prediction}")
                        
                        # Add probability if available
                        try:
                            proba = model.predict_proba(processed_input)[0]
                            classes = model.classes_
                            
                            # Create a dataframe for better display
                            proba_df = pd.DataFrame({
                                'Class': classes,
                                'Probability': proba
                            })
                            st.dataframe(proba_df.sort_values('Probability', ascending=False))
                        except:
                            pass
                    else:  # regression
                        st.write(f"Predicted Value: {prediction:.4f}")
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    # Return to home button
    if st.button("Return to Home"):
        st.session_state.page = "Data Upload"
        st.rerun()