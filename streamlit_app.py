import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

file1_path = "https://github.com/Najam-Mehdi/Insect-Prediction/blob/main/Insect_Caught.xlsx"
file2_path = "https://github.com/Najam-Mehdi/Insect-Prediction/blob/main/Temperature.xlsx"
df_insect = pd.read_excel(file1_path, engine='openpyxl')
df_temp = pd.read_excel(file2_path, engine='openpyxl')

df_insect = df_insect.drop_duplicates()
df_insect = df_insect.dropna()
df_insect['Date_Time'] = pd.to_datetime(df_insect['Date_Time'], format='%d.%m.%Y %H:%M:%S')

df_temp['Date_Time'] = pd.to_datetime(df_temp['Date_Time'], format='%d.%m.%Y %H:%M:%S')
df_merged = pd.merge(df_insect, df_temp, on='Date_Time', how='inner')

df_merged['Day'] = df_merged['Date_Time'].dt.day
df_merged['Month'] = df_merged['Date_Time'].dt.month
df_merged['Year'] = df_merged['Date_Time'].dt.year
df_merged['DayOfWeek'] = df_merged['Date_Time'].dt.weekday
df_merged['Hour'] = df_merged['Date_Time'].dt.hour
df_merged['Minute'] = df_merged['Date_Time'].dt.minute
df_merged['WeekOfYear'] = df_merged['Date_Time'].dt.isocalendar().week
df_merged['Prev_Num_Insects'] = df_merged['Number_of_Insects'].shift(1)
df_merged['Prev_Temperature'] = df_merged['Mean_Temperature'].shift(1)
df_merged['Prev_Humidity'] = df_merged['Mean_Humidity'].shift(1)
df_merged['Temp_Delta'] = df_merged['Temperature_High'] - df_merged['Temperature_Low']
df_merged['Rolling_Temperature'] = df_merged['Mean_Temperature'].rolling(window=3).mean()
df_merged['Rolling_Humidity'] = df_merged['Mean_Humidity'].rolling(window=3).mean()

df_cleaned = df_merged.dropna(subset=['Mean_Temperature', 'Temperature_Low', 'Temperature_High',
                                      'Mean_Humidity', 'Prev_Num_Insects', 'Prev_Temperature',
                                      'Prev_Humidity', 'Temp_Delta', 'Rolling_Temperature', 
                                      'Rolling_Humidity', 'Number_of_Insects', 'New_Catches'])

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a Page", ["EDA", "Modeling", "Prediction"])

if page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    if st.button("No. of Insects"):
     df_merged['Date'] = pd.to_datetime(df_merged[['Year', 'Month', 'Day']])
     df_daily = df_merged.groupby('Date').agg({
        'Number_of_Insects': 'sum',
        'New_Catches': 'sum',
        'Mean_Temperature': 'mean',
        'Mean_Humidity': 'mean'
     }).reset_index()

     st.subheader("Number of Insects Caught - Daily Aggregation")
     fig, ax = plt.subplots(figsize=(12, 6))
     ax.plot(df_daily['Date'], df_daily['Number_of_Insects'], label='Number of Insects', color='tab:red')
     ax.set_title("Number of Insects Caught - Daily Aggregation")
     ax.set_xlabel("Date")
     ax.set_ylabel("Number of Insects")
     ax.tick_params(axis='x', rotation=45)
     st.pyplot(fig)

    if st.button("No. of Catches"):
     st.subheader("New Insect Catches - Daily Aggregation")
     df_merged['Date'] = pd.to_datetime(df_merged[['Year', 'Month', 'Day']])
     df_daily = df_merged.groupby('Date').agg({
        'Number_of_Insects': 'sum',
        'New_Catches': 'sum',
        'Mean_Temperature': 'mean',
        'Mean_Humidity': 'mean'
     }).reset_index()
    
     fig, ax = plt.subplots(figsize=(12, 6))
     ax.plot(df_daily['Date'], df_daily['New_Catches'], label='New Catches', color='tab:blue')
     ax.set_title("New Insect Catches - Daily Aggregation")
     ax.set_xlabel("Date")
     ax.set_ylabel("New Catches")
     ax.tick_params(axis='x', rotation=45)
     st.pyplot(fig)
    
    if st.button("Correlation"):
     st.subheader("Correlation Matrix Between Pest Counts and Weather Variables")
     correlation_matrix = df_merged[['Number_of_Insects', 'Mean_Temperature', 'Mean_Humidity']].corr()
     fig, ax = plt.subplots(figsize=(8, 6))
     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
     ax.set_title("Correlation Matrix Between Pest Counts and Weather Variables")
     st.pyplot(fig)

    if st.button("Distribution"):
     st.subheader("Distribution of New Insect Catches")
     fig, ax = plt.subplots(figsize=(10, 6))
     sns.histplot(df_merged['New_Catches'], kde=True, bins=20, ax=ax)
     ax.set_title("Distribution of New Insect Catches")
     ax.set_xlabel("Number of New Catches")
     ax.set_ylabel("Frequency")
     st.pyplot(fig)

elif page == "Modeling":
    st.title("Model Evaluation")
    
    if st.button("Regression"):
        X_reg = df_cleaned[['Mean_Temperature', 'Temperature_Low', 'Temperature_High', 'Mean_Humidity',
                            'Prev_Num_Insects', 'Prev_Temperature', 'Prev_Humidity', 'Temp_Delta',
                            'Rolling_Temperature', 'Rolling_Humidity']]
        y_reg = df_cleaned['Number_of_Insects']
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_reg_scaled = scaler.fit_transform(X_train_reg)
        X_test_reg_scaled = scaler.transform(X_test_reg)
        
        rf_reg = RandomForestRegressor()
        rf_reg.fit(X_train_reg_scaled, y_train_reg)
        y_pred_reg = rf_reg.predict(X_test_reg_scaled)
        
        rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        r2 = r2_score(y_test_reg, y_pred_reg)
        
        st.subheader("Regression Model Evaluation")
        st.write(f'RMSE: {rmse}')
        st.write(f'MAE: {mae}')
        st.write(f'R2: {r2}')
    
    if st.button("Classification"):
        X_clf = df_cleaned[['Mean_Temperature', 'Temperature_Low', 'Temperature_High', 'Mean_Humidity',
                            'Prev_Num_Insects', 'Prev_Temperature', 'Prev_Humidity', 'Temp_Delta',
                            'Rolling_Temperature', 'Rolling_Humidity']]
        y_clf = (df_cleaned['New_Catches'] > 0).astype(int)
        
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_clf_scaled = scaler.fit_transform(X_train_clf)
        X_test_clf_scaled = scaler.transform(X_test_clf)
        
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train_clf_scaled, y_train_clf)
        y_pred_clf = rf_clf.predict(X_test_clf_scaled)
        
        accuracy = accuracy_score(y_test_clf, y_pred_clf)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_clf, y_pred_clf, average='binary')
        
        st.subheader("Classification Model Evaluation")
        st.write(f'Accuracy: {accuracy}')
        st.write(f'Precision: {precision}')
        st.write(f'Recall: {recall}')
        st.write(f'F1-Score: {f1}')

elif page == "Prediction":
    st.title("Prediction")
    
    model_type = st.radio("Select Model Type", ["Regression", "Classification"])
    
    st.subheader("Input Features")
    mean_temp = st.number_input("Mean Temperature")
    temp_low = st.number_input("Temperature Low")
    temp_high = st.number_input("Temperature High")
    mean_humidity = st.number_input("Mean Humidity")
    prev_num_insects = st.number_input("Previous Number of Insects", value=0)
    prev_temp = st.number_input("Previous Temperature")
    prev_humidity = st.number_input("Previous Humidity")
    temp_delta = temp_high - temp_low
    rolling_temp = st.number_input("Rolling Mean Temperature")
    rolling_humidity = st.number_input("Rolling Mean Humidity")
    
    input_features = np.array([[mean_temp, temp_low, temp_high, mean_humidity,
                                prev_num_insects, prev_temp, prev_humidity,
                                temp_delta, rolling_temp, rolling_humidity]])
    
    scaler = StandardScaler()
    df_features = df_cleaned[['Mean_Temperature', 'Temperature_Low', 'Temperature_High', 
                              'Mean_Humidity', 'Prev_Num_Insects', 'Prev_Temperature', 
                              'Prev_Humidity', 'Temp_Delta', 'Rolling_Temperature', 
                              'Rolling_Humidity']]
    scaler.fit(df_features)
    input_scaled = scaler.transform(input_features)
    
    if model_type == "Regression":
        X_reg = df_features
        y_reg = df_cleaned['Number_of_Insects']
        rf_reg = RandomForestRegressor()
        rf_reg.fit(X_reg, y_reg)
        prediction = rf_reg.predict(input_scaled)
        st.write(f"Predicted Number of Insects: {prediction[0]:.2f}")
    
    elif model_type == "Classification":
        X_clf = df_features
        y_clf = (df_cleaned['New_Catches'] > 0).astype(int)
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_clf, y_clf)
        prediction = rf_clf.predict(input_scaled)
        prediction_prob = rf_clf.predict_proba(input_scaled)
        st.write(f"Predicted Class: {prediction[0]} (0: No Catch, 1: Catch)")
        st.write(f"Prediction Probability: {prediction_prob[0][1]:.2%}")
