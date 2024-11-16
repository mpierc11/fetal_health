# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App') 

# Display an image of penguins
st.image('fetal_health_image.gif', width = 400)
st.caption("Utilize our advanced Machine Learning Application to predict fetal health classifications.")


# Load the pre-trained model from the pickle file
dt_pickle = open('decision_tree_fetal_health.pickle', 'rb') 
rf_pickle = open('random_forest_fetal_health.pickle', 'rb') 
ab_pickle = open('adaboost_fetal_health.pickle', 'rb') 
clf_dt = pickle.load(dt_pickle) 
dt_pickle.close()
clf_rf = pickle.load(rf_pickle) 
rf_pickle.close()
clf_ab = pickle.load(ab_pickle) 
ab_pickle.close()


# Load the default dataset - for dummy encoding (properly)
default_df = pd.read_csv('fetal_health.csv')

# Create a sidebar for input collection
st.sidebar.header("Fetal Health Features Input")


with st.sidebar:
    user_csv = st.file_uploader("Upload your CSV file here")
    st.header('Sample Data Format for Upload')
    st.dataframe(default_df.head(5))
  
model = st.sidebar.radio('Model', options = ['Decision Tree', 'Random Forest'])


# Process the uploaded CSV file
if user_csv is not None:
    user_df = pd.read_csv(user_csv)  # Load the uploaded CSV into a DataFrame

    # Reset the index of the uploaded file and default_df for consistency
    user_df.reset_index(drop=True, inplace=True)  # Reset index of uploaded CSV
    # default_df.reset_index(drop=True, inplace=True)  # Reset index of default_df
   
    # Define the required feature columns that the model expects
    required_columns = [
        'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 
        'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 
        'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 
        'percentage_of_time_with_abnormal_long_term_variability', 
        'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 
        'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 
        'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 
        'histogram_tendency'
    ]

    # Find the intersection of required columns and uploaded columns
    available_columns = [col for col in required_columns if col in user_df.columns]

    # Check if we have any available columns for prediction
    if available_columns:
        # Select the model based on user's choice and make predictions
        if model == 'Decision Tree':
            predictions = clf_dt.predict(user_df[available_columns])  # Predict class
            prob_predictions = clf_dt.predict_proba(user_df[available_columns])  # Predict probabilities
        elif model == 'Random Forest':
            predictions = clf_rf.predict(user_df[available_columns])  # Predict class
            prob_predictions = clf_rf.predict_proba(user_df[available_columns])  # Predict probabilities

        # Add the predictions and probability columns to the DataFrame
        user_df['Predicted Class'] = predictions
        user_df['Prediction Probability'] = prob_predictions.max(axis=1)  # Taking the max probability

        # Display the updated DataFrame with predictions
        st.header('Predictions with Probabilities')
        st.dataframe(user_df)
    else:
        st.warning("The uploaded CSV doesn't contain any required columns for prediction. The model was run with whatever features were available.")