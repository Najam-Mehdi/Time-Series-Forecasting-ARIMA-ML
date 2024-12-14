import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import urllib


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

try:
    df_temp = load_data("https://raw.githubusercontent.com/Najam-Mehdi/Insect-Prediction/refs/heads/main/docs/Temperature.csv")
    df_insect = load_data("https://raw.githubusercontent.com/Najam-Mehdi/Insect-Prediction/refs/heads/main/docs/Insect_Caught.csv")
except urllib.error.URLError as e:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    df_temp = load_data("https://raw.githubusercontent.com/Najam-Mehdi/Insect-Prediction/refs/heads/main/docs/Temperature.csv")
    df_insect = load_data("https://raw.githubusercontent.com/Najam-Mehdi/Insect-Prediction/refs/heads/main/docs/Insect_Caught.csv")


# Drop duplicates and null values
def data_cleaning(data):
    data = data.drop_duplicates()
    data = data.dropna()
    return data

df_insect = data_cleaning(df_insect)
df_temp = data_cleaning(df_temp)

def date_parsing(df):
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%d.%m.%Y %H:%M:%S')
    df.drop_duplicates(subset='Date_Time', keep='first')
    df['Time'] = df['Date_Time'].dt.time
    return df

df_insect = date_parsing(df_insect)
df_temp = date_parsing(df_temp)
# Issue with same date time value with different other values. so dropping the duplicate datetime and keeping the first values.

df_merged = pd.merge(df_insect, df_temp, on='Date_Time', how='inner')

def date_preprocessing(df):
    df['Day'] = df['Date_Time'].dt.day
    df['Month'] = df['Date_Time'].dt.month
    df['Year'] = df['Date_Time'].dt.year
    df['DayOfWeek'] = df['Date_Time'].dt.weekday
    df['Hour'] = df['Date_Time'].dt.hour
    df['Minute'] = df['Date_Time'].dt.minute
    df['WeekOfYear'] = df['Date_Time'].dt.isocalendar().week
    return df

df_merged = date_preprocessing(df_merged)
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

@st.cache_data
def plot_correlation(df, cols: list[str]):
    correlation_matrix = df[cols].corr()

    # Create a Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )

    # Update layout
    fig.update_layout(width=800, height=700)
    return fig


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

        # Create an interactive Plotly line chart
        fig = px.line(
            df_daily,
            x='Date',
            y='Number_of_Insects',
            title="Number of Insects Caught - Daily Aggregation",
            labels={'Number_of_Insects': 'Number of Insects', 'Date': 'Date'},
            line_shape='spline',  # Smoother line
            markers=True  # Add markers to data points
        )

        # Customize the layout for better visuals
        fig.update_layout(
            title={'x': 0.5},  # Center the title
            xaxis_title="Date",
            yaxis_title="Number of Insects",
            xaxis=dict(showgrid=True, tickangle=45),
            yaxis=dict(showgrid=True),
            template="plotly_white"  # Use a clean theme
        )

        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    if st.button("No. of Catches"):
        # Ensure the Date column is parsed correctly
        df_merged['Date'] = pd.to_datetime(df_merged[['Year', 'Month', 'Day']])

        # Aggregate daily data
        df_daily = df_merged.groupby('Date').agg({
            'Number_of_Insects': 'sum',
            'New_Catches': 'sum',
            'Mean_Temperature': 'mean',
            'Mean_Humidity': 'mean'
        }).reset_index()

        fig_new_catches = px.line(
            df_daily,
            x='Date',
            y='New_Catches',
            title="New Insect Catches - Daily Aggregation",
            labels={'New_Catches': 'New Catches', 'Date': 'Date'},
            line_shape='spline',
            markers=True
        )
        fig_new_catches.update_layout(
            title={'x': 0.5},
            xaxis_title="Date",
            yaxis_title="New Catches",
            xaxis=dict(showgrid=True, tickangle=45),
            yaxis=dict(showgrid=True),
            template="plotly_white"
        )
        st.plotly_chart(fig_new_catches, use_container_width=True)
        
    if st.button("Correlation"):
        st.subheader("Correlation Matrix Between Pest Counts and Weather Variables")
        cols = ["Number_of_Insects", "Mean_Temperature", "Mean_Humidity"]
        fig = plot_correlation(df_merged, cols)
        st.plotly_chart(fig)
        

    if st.button("Distribution"):
        st.subheader("Distribution of New Insect Catches")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df_merged['New_Catches'], kde=True, bins=20, ax=ax)
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
