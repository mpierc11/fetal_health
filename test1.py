# Get Predictions + Pics
if model == 'Decision Tree':
  dt_prediction = clf_dt.predict([[ 'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 
        'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 
        'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 
        'percentage_of_time_with_abnormal_long_term_variability', 
        'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 
        'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 
        'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 
        'histogram_tendency']])
  prediction_dt = dt_prediction[0]
  st.subheader("Predicting Your Mobile's Price")
  st.success(f'We predict your mobile is in the price range of {prediction_dt}.')   
   

  #  Probability estimation
  features= [[ 'baseline_value', 'accelerations', 'fetal_movement', 'uterine_contractions', 
        'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 
        'abnormal_short_term_variability', 'mean_value_of_short_term_variability', 
        'percentage_of_time_with_abnormal_long_term_variability', 
        'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min', 
        'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 
        'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 
        'histogram_tendency']]
  
  dt_proba = clf_dt.predict_proba(features)[0]
  st.write('Probability Estimation: ', dt_proba.max())
  
  # Showing additional items in tabs
  st.subheader("Model Performance and Insights")
  tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Feature Importance Visualization
  with tab1:
        st.write("### Feature Importance")
        st.image('feature_imp_dt.svg')
        st.caption("Features used in this prediction are ranked by relative importance.")

    # Tab 2: Confusion Matrix
  with tab2:
        st.write("### Confusion Matrix")
        st.image('price_dt_confusion_mat.svg')
        st.caption("Confusion Matrix of model predictions.")

    # Tab 3: Classification Report
  with tab3:
        st.write("### Classification Report")
        report_df = pd.read_csv('price_dt_class_report.csv', index_col = 0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
        st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")


# elif model == 'Random Forest':
#   rf_prediction = clf_rf.predict([[battery_power, clock_speed, fc, int_memory, m_dep,
#         mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h,
#         sc_w, talk_time, blue_No, blue_Yes, dual_sim_No,
#         dual_sim_Yes, four_g_No, four_g_Yes, three_g_No, three_g_Yes,
#         touch_screen_No, touch_screen_Yes, wifi_No, wifi_Yes]])
#   prediction_rf = rf_prediction[0]
#   st.subheader("Predicting Your Mobile's Price")
#   st.success(f'We predict your mobile is in the price range of {prediction_rf}.')

# #  Probability estimation
#   features= [[battery_power, clock_speed, fc, int_memory, m_dep,
#        mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h,
#        sc_w, talk_time, blue_No, blue_Yes, dual_sim_No,
#        dual_sim_Yes, four_g_No, four_g_Yes, three_g_No, three_g_Yes,
#        touch_screen_No, touch_screen_Yes, wifi_No, wifi_Yes]]
  
#   rf_proba = clf_rf.predict_proba(features)[0]
#   st.write('Probability Estimation: ', rf_proba.max())



#     # Showing additional items in tabs
#   st.subheader("Model Performance and Insights")
#   tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

#     # Tab 1: Feature Importance Visualization
#   with tab1:
#         st.write("### Feature Importance")
#         st.image('feature_imp_rf.svg')
#         st.caption("Features used in this prediction are ranked by relative importance.")

#     # Tab 2: Confusion Matrix
#   with tab2:
#         st.write("### Confusion Matrix")
#         st.image('price_rf_confusion_matrix.svg')
#         st.caption("Confusion Matrix of model predictions.")

#     # Tab 3: Classification Report
#   with tab3:
#         st.write("### Classification Report")
#         report_df = pd.read_csv('price_rf_class_report.csv', index_col = 0).transpose()
#         st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
#         st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each species.")




