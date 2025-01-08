import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from utils import *
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="BI Streamlit App",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Page 1: Data Load & Review", "Page 2: Data Preprocessing", "Page 3: Descriptive Statistics", "Page 4: Statistical Analysis of Variables", "Page 5: Data Visualization", "Page 6: Fixed Partitioning", "Page 7: Statistical Forecast Methods", "Page 8: ARIMA", "Page 9: SARIMA", "Page 10: Machine Learning Models", "Page 11: Model Comparison", "Page 12: Forecasting with the Best Model"])

# Home Page
if page == "Home":
    st.title("📚 Welcome to the Information Systems & Business Intelligence - Streamlit App!")
    st.write("""
    This Project Will Tell You That Why You Are Ugly.
    """)
    st.image(
        "https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-lighttext.png",
        caption="Streamlit Logo",
        use_container_width=True,  # Updated parameter
    )

# Page 1: Buttons
elif page == "Page 1: Data Load & Review":
    st.title("🎯 Data Load & Review")
    st.write("This page loads and reviews the initial data.")

    # Button examples
    if st.button("Load Data"):
        root_path = "https://raw.githubusercontent.com/Najam-Mehdi/Insect-Prediction/refs/heads/main/data/"
        def load_data(filepath):
            data = pd.read_csv(filepath)
            return data
        try:
            df_temp = load_data(root_path + "temperature.csv")
            df_ins = load_data(root_path + "insect_Caught.csv")
        except urllib.error.URLError as e:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            df_temp = load_data(root_path + "temperature.csv")
            df_ins = load_data(root_path + "insect_Caught.csv")
        
        # Store data in session state
        st.session_state['df_ins'] = df_ins
        st.session_state['df_temp'] = df_temp
        st.write("Data Loaded Successfully")

    # Review Data Button
    if "df_ins" in st.session_state and "df_temp" in st.session_state:
        if st.button("Review Data"):
            st.write("Insects Data Sample:")
            st.write(st.session_state['df_ins'].sample(5))
            st.write("Temperature Data Sample:")
            st.write(st.session_state['df_temp'].sample(5))
    else:
        st.warning("Please load the data first.")

# Page 2: Info
elif page == "Page 2: Data Preprocessing":
    st.title("💡 Data Preprocessing")
    st.write("This page performs preprocessing.")

    if "df_ins" in st.session_state and "df_temp" in st.session_state:
        if st.button("Preprocess Data"):
            df_ins = st.session_state['df_ins']
            df_temp = st.session_state['df_temp']

            df_ins.dropna(inplace=True)
            df_ins.drop_duplicates(inplace=True)
            df_ins['Date'] = pd.to_datetime(df_ins['Date'], format='mixed', dayfirst=True)

            df_temp.dropna(inplace=True)
            df_temp.drop_duplicates(inplace=True)
            df_temp['Mean_Temperature'] = df_temp['Mean_Temperature'].str.replace(',', '.').astype(float)
            df_temp['Temperature_High'] = df_temp['Temperature_High'].str.replace(',', '.').astype(float)
            df_temp['Temperature_Low'] = df_temp['Temperature_Low'].str.replace(',', '.').astype(float)
            df_temp['Mean_Humidity'] = df_temp['Mean_Humidity'].str.replace(',', '.').astype(float)
            df_temp['Date'] = pd.to_datetime(df_temp['Date'], format='mixed', dayfirst=True)

            df_merged = pd.merge(df_ins, df_temp, on='Date', how='left')
            st.session_state['df_merged'] = df_merged
            st.write("Data Review:")
            st.write(data_review(df_merged))  # Display the output of data_review
            st.write("Data Preprocessed Successfully.")

            df_grouped = df_merged.groupby('Date').agg({
                'Number_of_Insects': 'sum',
                'New_Catches': 'sum',
                'Mean_Temperature': 'mean',
                'Temperature_High': 'mean',
                'Temperature_Low': 'mean',
                'Mean_Humidity': 'mean'
            }).reset_index()
            st.session_state['df_grouped'] = df_grouped
    else:
        st.warning("Please load the data first on the Data Load & Review page.")

elif page == "Page 3: Descriptive Statistics":
    st.title("Descriptive Statistics")
    if "df_grouped" in st.session_state:
        if st.button("Describe Data"):
            st.write(st.session_state['df_grouped'].describe())
    else:
        st.warning("Please preprocess the data first.")

elif page == "Page 4: Statistical Analysis of Variables":
    st.title("Statistical Analysis of Variables")

    if "df_grouped" in st.session_state:
        df_grouped = st.session_state['df_grouped']

        analysis_buttons = {
            "Univariate Analysis": False,
            "Correlation Analysis": False,
            "ANOVA Test - Temperature": False,
            "ANOVA Test - Humidity": False,
            "Box Plot": False
        }

        for key in analysis_buttons.keys():
            analysis_buttons[key] = st.button(key)

        if analysis_buttons["Univariate Analysis"]:
            cols = ['Number_of_Insects', 'New_Catches', 'Mean_Temperature', 'Temperature_Low', 'Temperature_High',
                    'Mean_Humidity']
            fig, axes = plt.subplots(2, 3, figsize=(10, 6))

            for i, ax in enumerate(axes.flat):
                sns.histplot(data=df_grouped, x=cols[i], kde=True, ax=ax, edgecolor=".3")

                ax.set_title(f'{cols[i]}', pad=10, fontsize=8)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticklabels("")
                ax.set_yticklabels("")
                ax.tick_params(axis='both', which='both', bottom=False, left=False)
            st.pyplot(fig)
            st.success("Observation: It appears that the data in the features do not follow a normal distribution. To assess this further, we will visualize the Q-Q plots of the features.")

            fig, axes = plt.subplots(2, 3, figsize=(10, 6))

            for i, ax in enumerate(axes.flat):
                sm.qqplot(data=df_grouped[cols[i]], ax=ax, line="45")
                ax.set_title(f'{cols[i]}', pad=10, fontsize=8)
                ax.set_xticklabels("")
                ax.set_yticklabels("")
                ax.tick_params(axis='both', which='both', bottom=False, left=False)
                ax.set_xlabel("")
                ax.set_ylabel("")
            st.pyplot(fig)
            st.success("Observation: The data features show deviations from a normal distribution, as indicated by the Q-Q plots. To make the features more comparable, we will standardize the values")

        if analysis_buttons["Correlation Analysis"]:
            cols = ['Number_of_Insects', 'New_Catches', 'Mean_Temperature', 'Temperature_Low', 'Temperature_High',
                    'Mean_Humidity']
            cmap = mcolors.LinearSegmentedColormap.from_list('custom_blue', ['#FFFFFF', '#84ACC8'])

            corr_matrix = round(df_grouped[cols].corr(), 2)
            st.write("Correlation Matrix:")
            st.write(corr_matrix)  # Display the correlation matrix as a table
            st.success("Observation: The linear relationships between the target variable and the predictors vary in strength. Humidity shows no significant relationship with the target variable, while temperature exhibits a moderate positive relationship. The strongest relationship is observed between 'Number_of_Insects' which is strong and positive. We will perform ANOVA test to explore the effects Mean Temperature and Mean Humidity on New_Catches through group-based analysis.")

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap=cmap, linewidths=0.5, annot_kws={"size": 7}, cbar=False, ax=ax)
            st.pyplot(fig)
            st.success("Observation: Since there seems to be multicollinearity between the three temperature features, we will remove the high and low temperature features.")

        if analysis_buttons["ANOVA Test - Temperature"]:
            bins = [20.345, 23, 27, 30.30725]
            labels = ['Low', 'Medium', 'High']

            df_grouped['Temperature_Bin'] = pd.cut(df_grouped['Mean_Temperature'], bins=bins, labels=labels,
                                                   include_lowest=True)
            groups = [group['New_Catches'].values for _, group in df_grouped.groupby('Temperature_Bin')]
            f_stat, p_value = f_oneway(*groups)

            st.write(f"F-statistic: {f_stat}")
            st.write(f"P-value: {p_value}")
            df_grouped.drop(columns='Temperature_Bin', inplace=True)
            st.write("Data After Dropping 'Temperature_Bin':")
            st.write(df_grouped)  # Display the table to ensure visibility
            st.success("Observation: The results suggest that Mean_Temperature (grouped into Low, Medium, High) does not significantly impact New_Catches.")

        if analysis_buttons["ANOVA Test - Humidity"]:
            bins = [44.415, 59, 74, 88.884]
            labels = ['Low', 'Medium', 'High']

            df_grouped['Humidity_Bin'] = pd.cut(df_grouped['Mean_Humidity'], bins=bins, labels=labels,
                                                include_lowest=True)
            groups = [group['New_Catches'].values for _, group in df_grouped.groupby('Humidity_Bin')]
            f_stat, p_value = f_oneway(*groups)

            st.write(f"F-statistic: {f_stat}")
            st.write(f"P-value: {p_value}")
            df_grouped.drop(columns=['Humidity_Bin'], inplace=True)
            st.write("Data After Dropping 'Humidity_Bin':")
            st.write(df_grouped)  # Display the table to ensure visibility
            st.success("Observation: The results suggest that Mean_Humidity (grouped into Low, Medium, High) significantly impacts New_Catches, as indicated by the p-value of 0.0084, which is below the 0.05 significance threshold. This suggests that there is a meaningful difference in the mean number of New_Catches between the different humidity categories.")

        if analysis_buttons["Box Plot"]:
            features = ['Number_of_Insects', 'New_Catches', 'Mean_Temperature', 'Mean_Humidity']

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for i, feature in enumerate(features):
                sns.boxplot(data=df_grouped, y=feature, ax=axes[i // 2, i % 2])
                axes[i // 2, i % 2].set_title(f'Boxplot of {feature}')
            st.pyplot(fig)
            st.success("Observations: There are outliers in all the features.")

    else:
        st.warning("Please preprocess the data first.")

elif page == "Page 5: Data Visualization":
    st.title("Data Visualization")
    st.write("Components of the Time-Series")

    if "df_grouped" in st.session_state:
        df_grouped = st.session_state['df_grouped']

        if st.button("Components of New Catches"):
            # Plot for Components of New Catches
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_grouped['Date'], df_grouped['New_Catches'], label='New Catches')
            ax.set_title('Components of New Catches')
            ax.set_xlabel('Date')
            ax.set_ylabel('New Catches')
            st.pyplot(fig)
            st.success("Observation: The data shows seasonality with peaks and troughs at regular intervals, along with a potential upward trend towards the end of the period. We will decompose the series into individual components to verify that.")

        if st.button("New Catches"):
            result = seasonal_decompose(df_grouped['New_Catches'], model='additive', period=12)  # Adjust 'period' based on your data frequency
            result.plot()
            plt.tight_layout()
            st.pyplot()
            st.success("Observation: Trend: Shows a gradual upward movement, indicating an increase in catches over time. Seasonal: Reveals regular cyclical patterns with consistent peaks and troughs. Residual: Displays random fluctuations around zero, suggesting the model has captured most patterns.")
    else:
        st.warning("Please preprocess the data first.")


elif page == "Page 6: Fixed Partitioning":
    st.title("📊 Fixed Partitioning")
    st.write("This page demonstrates fixed partitioning of the data into training and validation datasets.")

    if "df_grouped" in st.session_state:
        df_grouped = st.session_state['df_grouped']

        if st.button("Do Partition"):
            # Perform fixed partitioning
            n_train_samples = 40
            X_train = df_grouped.iloc[0:n_train_samples]
            X_val = df_grouped.iloc[n_train_samples:]

            st.write(f"Training Set Shape: {X_train.shape}")
            st.write(f"Validation Set Shape: {X_val.shape}")

            # Plotting Training and Validation datasets
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # Training dataset
            ax[0].plot(X_train['Date'], X_train['Mean_Temperature'], label='Mean Temperature', color='blue')
            ax[0].plot(X_train['Date'], X_train['Mean_Humidity'], label='Mean Humidity', color='orange')
            ax[0].set_title("Training Dataset (80%)")
            ax[0].legend()

            # Validation dataset
            ax[1].plot(X_val['Date'], X_val['Mean_Temperature'], label='Mean Temperature', color='blue')
            ax[1].plot(X_val['Date'], X_val['Mean_Humidity'], label='Mean Humidity', color='orange')
            ax[1].set_title("Validation Dataset (20%)")
            ax[1].legend()

            st.pyplot(fig)
    else:
        st.warning("Please preprocess the data first on the Data Preprocessing page.")

elif page == "Page 7: Statistical Forecast Methods":
    st.title("📈 Statistical Forecast Methods")
    st.write("Naive Forecast: A simple, baseline method for making predictions. It assumes that the future will look like the past.")
    st.write("Moving Average Forecast: A forecasting technique that predicts future values by averaging a fixed number of past observations to smooth out short-term fluctuations and highlight longer-term trends.")
    st.write("Differenced Moving Average Forecast: A time series transformation technique that subtracts the current observation from a previous one to achieve stationarity by removing trends and seasonality.")

    if "df_grouped" in st.session_state:
        df_grouped = st.session_state['df_grouped']
        n_train_samples = 40  # Ensure this is consistent with earlier usage
        X_val = df_grouped.iloc[n_train_samples:]

        # Initialize shared variables in session_state if not already set
        if "naive_forecast" not in st.session_state:
            st.session_state["naive_forecast"] = df_grouped['New_Catches'].iloc[n_train_samples - 1:-1]
        if "nf_mae" not in st.session_state:
            st.session_state["nf_mae"] = mean_absolute_error(X_val['New_Catches'], st.session_state["naive_forecast"])

        # Naive Forecast
        if st.button("Show Naive Forecast"):
            naive_forecast = st.session_state["naive_forecast"]
            nf_mae = st.session_state["nf_mae"]

            time_step = 8
            st.write(f"Ground truth at time step {time_step}: {X_val['New_Catches'].iloc[time_step]}")
            st.write(f"Prediction at time step {time_step + 1}: {naive_forecast.iloc[time_step + 1]}")
            st.write(f"Mean Absolute Error for Naive Forecast: {nf_mae}")

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_series(X_val['New_Catches'].iloc[1:10], X_val['New_Catches'].iloc[1:10], label='Ground Truth', title='Naive Forecast')
            plot_series(naive_forecast.iloc[0:10], naive_forecast.iloc[0:10], label='Predictions', title='Naive Forecast')
            st.pyplot(fig)

            st.success("Observation: It is clear that the predicted values are shifted by one time-step ahead of the actual values.")

        # Moving Average Forecast
        if st.button("Show Moving Average Forecast"):
            window_size = 3
            moving_avg = df_grouped['New_Catches'].rolling(window=window_size).mean()
            moving_avg = moving_avg.iloc[n_train_samples:]

            # Compute MAE
            ma_mae = mean_absolute_error(X_val['New_Catches'], moving_avg)
            st.write(f"Mean Absolute Error for Moving Average Forecast: {ma_mae}")
            st.session_state['ma_mae'] = ma_mae

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_series(X_val['New_Catches'].iloc[0:10], X_val['New_Catches'].iloc[0:10], label='Ground Truth', title='Moving Average Forecast')
            plot_series(moving_avg.iloc[0:10], moving_avg.iloc[0:10], label='Predictions', title='Moving Average Forecast')
            st.pyplot(fig)

        # Differenced Moving Average Forecast
        # Differenced Moving Average Forecast
        if st.button("Show Differenced Moving Average Forecast"):
            periods = 2
            diff_series = df_grouped['New_Catches'].diff(periods=periods).iloc[n_train_samples:]

            window_size = 3
            diff_moving_avg = diff_series.rolling(window=window_size).mean()

            # Display null value count before filling
            st.write(f"Null values after creating the moving average: {diff_moving_avg.isnull().sum()}")

            # Backfill null values
            diff_moving_avg = diff_moving_avg.bfill()

            # Display null value count after filling
            st.write(f"Null values after backfilling: {diff_moving_avg.isnull().sum()}")

            # Compute MAE
            ma_mae_diff = mean_absolute_error(X_val['New_Catches'], diff_moving_avg)
            st.write(f"Mean Absolute Error for Differenced Moving Average Forecast: {ma_mae_diff}")
            st.session_state['ma_mae_diff'] = ma_mae_diff

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_series(X_val['New_Catches'].iloc[0:10], X_val['New_Catches'].iloc[0:10], label='Ground Truth',
                        title='Differenced Moving Average Forecast')
            plot_series(diff_moving_avg.iloc[0:10], diff_moving_avg.iloc[0:10], label='Predictions',
                        title='Differenced Moving Average Forecast')
            st.pyplot(fig)

    else:
        st.warning("Please preprocess the data first on the Data Preprocessing page.")

elif page == "Page 8: ARIMA":
    st.title("📈 ARIMA Model Analysis")
    st.write("This page demonstrates ARIMA model analysis with various steps.")

    if "df_grouped" in st.session_state:
        df_grouped = st.session_state['df_grouped']

        # Button 1: Show Head
        if st.button("Show Head"):
            st.write(df_grouped.head())

        # Button 2: Show ADF Stats & P-Value
        if st.button("Show ADF Stats & P-Value"):
            df_grouped['Date'] = pd.to_datetime(df_grouped['Date'])
            df_grouped.set_index('Date', inplace=True)
            result = adfuller(df_grouped['New_Catches'].dropna())
            st.write(f"ADF Statistic: {result[0]}")
            st.write(f"p-value: {result[1]}")

            st.success("Observation: The p-value is greater than 0.05 (0.6579), the time series is non-stationary.")

        # Button 3: Show Improved ADF Stats & P-Value
        if st.button("Show Improved ADF Stats & P-Value"):
            df_grouped['New_Catches_diff'] = df_grouped['New_Catches'] - df_grouped['New_Catches'].shift(1)
            df_grouped['New_Catches_diff'].dropna(inplace=True)
            result = adfuller(df_grouped['New_Catches_diff'].dropna())
            st.write(f"ADF Statistic: {result[0]}")
            st.write(f"p-value: {result[1]}")

            st.success("Observation: The p-value is now very small, indicating that the differenced series is stationary.")
            st.success("We will use auto-Arima to get the best values for p, d, and q. p is about using previous data to predict the next value. d is about making the data more stable by removing trends. q is about correcting the predictions based on past mistakes.")

        # Button 4: Run ARIMA Model
        if st.button("Run ARIMA Model"):
            model = auto_arima(df_grouped['New_Catches'].dropna(), seasonal=False, stepwise=True, trace=True)
            model = ARIMA(df_grouped['New_Catches'], order=(1, 0, 2))
            fitted_model = model.fit()

            # In-sample prediction
            predictions = fitted_model.predict(start=0, end=len(df_grouped['New_Catches'])-1)

            # Plot the original and predicted values
            plt.figure(figsize=(10, 6))
            plt.plot(df_grouped['New_Catches'], label='Original')
            plt.plot(df_grouped.index, predictions, label='Predicted', color='red')
            plt.legend()
            plt.title('Original vs In-sample Predictions')
            st.pyplot(plt)

            arima_mae = mean_absolute_error(df_grouped['New_Catches'], predictions)
            st.write(f"Mean Absolute Error for ARIMA: {arima_mae}")

            st.session_state['arima_mae'] = arima_mae

    else:
        st.warning("Please preprocess the data first on the Data Preprocessing page.")

elif page == "Page 9: SARIMA":
    st.title("📈 SARIMA Model Analysis")
    st.write("This page demonstrates SARIMA model analysis with various steps.")

    if "df_grouped" in st.session_state:
        df_grouped = st.session_state['df_grouped']

        # Run SARIMA Model Button
        if st.button("Run SARIMA"):
            # Fit SARIMA Model
            sarima_model = auto_arima(
                df_grouped['New_Catches'],
                seasonal=True,
                m=7,  # 7 for weekly seasonality
                stepwise=True,
                trace=True
            )

            # Store the SARIMA model in session state
            st.session_state['sarima_model'] = sarima_model

            st.write(sarima_model.summary())
            st.success("SARIMA Model has been fitted successfully.")

        # Show Plot and MAE Button
        if st.button("Show Plot and MAE"):
            # Ensure the model is in session state before proceeding
            if 'sarima_model' in st.session_state:
                sarima_model = st.session_state['sarima_model']

                best_order = sarima_model.order
                best_seasonal_order = sarima_model.seasonal_order

                # Fit the SARIMAX model
                fitted_sarima = SARIMAX(
                    df_grouped['New_Catches'],
                    order=best_order,
                    seasonal_order=best_seasonal_order
                ).fit(disp=False)

                # In-sample predictions
                sarima_predictions = fitted_sarima.predict(start=0, end=len(df_grouped['New_Catches'])-1)

                st.session_state['fitted_sarima'] = fitted_sarima

                # Plot the original and predicted values
                plt.figure(figsize=(10, 6))
                plt.plot(df_grouped['New_Catches'], label='Original')
                plt.plot(df_grouped.index, sarima_predictions, label='SARIMA Predicted', color='orange')
                plt.legend()
                plt.title('Original vs SARIMA In-sample Predictions')
                st.pyplot(plt)

                # Calculate and display MAE for SARIMA
                sarima_mae = mean_absolute_error(df_grouped['New_Catches'], sarima_predictions)
                st.write(f"Mean Absolute Error (MAE) for SARIMA: {sarima_mae}")

                st.session_state['sarima_mae'] = sarima_mae

            else:
                st.error("The SARIMA model has not been fitted yet. Please run the model first.")
    else:
        st.warning("Please preprocess the data first on the Data Preprocessing page.")

elif page == "Page 10: Machine Learning Models":
    st.title("🤖 Machine Learning Models")
    st.write("Feature Engineering:\n\n Day of the Year: Identifies yearly patterns.\n\nDay of the Week: Captures weekly trends.\n\nWeek of the Year: Tracks seasonal changes by week.\n\nLag_1, Lag_2, Lag_3: Uses past catches to predict future ones.\n\nRolling Mean_7: Smooths data to highlight long-term trends.\n\nTemp_Humidity_Interaction: Shows how temperature and humidity together affect catches.")

    if "df_grouped" in st.session_state:
        df_grouped = st.session_state['df_grouped']

        # Button 1: Do Machine Learning
        if st.button("Do Machine Learning"):
            df_grouped = st.session_state['df_grouped']

            # Feature Engineering
            df_grouped['Day_of_Week'] = df_grouped.index.dayofweek
            df_grouped['Week_of_Year'] = df_grouped.index.isocalendar().week
            df_grouped['Lag_1'] = df_grouped['New_Catches'].shift(1)
            df_grouped['Lag_2'] = df_grouped['New_Catches'].shift(2)
            df_grouped['Lag_3'] = df_grouped['New_Catches'].shift(3)
            df_grouped['Rolling_Mean_2'] = df_grouped['New_Catches'].rolling(window=2).mean()
            df_grouped['Temp_Humidity_Interaction'] = df_grouped['Mean_Temperature'] * df_grouped['Mean_Humidity']

            # Preparing Data for Modeling
            X = df_grouped.drop(columns=['New_Catches'])
            y = df_grouped['New_Catches']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

            # Scaling the Features
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            # Model Training
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train_sc, y_train)

            # Predictions and MAE Calculation
            y_pred = model.predict(X_test_sc)
            rf_mae = mean_absolute_error(y_test, y_pred)

            st.session_state['rf_mae'] = rf_mae

            st.write(f"Random Forest Mean Absolute Error: {rf_mae}")

        # Button 2: FIT Data
        if st.button("FIT Data"):
            from prophet import Prophet

            df_grouped_reset = df_grouped.reset_index()
            prophet_data = df_grouped_reset[['Date', 'New_Catches']].rename(columns={'Date': 'ds', 'New_Catches': 'y'})
            st.write("prophet data head")
            st.write(prophet_data.head())
            m = Prophet()
            m.fit(prophet_data)
            st.session_state['m'] = m
            st.success("Prophet model fitted successfully.")

            st.session_state['prophet_data'] = prophet_data

        # Button 3: Show Future Dates
        if st.button("Show Future Dates"):
            m = st.session_state['m']

            future = m.make_future_dataframe(periods=10)
            st.write(future)

        # Button 4: Show Future Forecast
        if st.button("Show Future"):
            m = st.session_state['m']

            future = m.make_future_dataframe(periods=10)
            forecast = m.predict(future)

            st.write(forecast[['ds', 'yhat']])
            st.session_state['forecast'] = forecast

        # Button 5: Show MAE
        if st.button("Show MAE"):
            prophet_data = st.session_state['prophet_data']
            forecast = st.session_state['forecast']

            merged_data = pd.merge(prophet_data, forecast[['ds', 'yhat']], on='ds', how='left')
            prop_mae = mean_absolute_error(merged_data['y'], merged_data['yhat'])

            st.session_state['prop_mae'] = prop_mae

            st.write(f"Prophet Model Mean Absolute Error: {prop_mae}")

    else:
        st.warning("Please preprocess the data first on the Data Preprocessing page.")


elif page == "Page 11: Model Comparison":
    st.title("📊 Model Comparison")

    # Button: Model Comparison - MAE Values
    if st.button("Model Comparison - MAE Values"):
        nf_mae = st.session_state['nf_mae']
        ma_mae = st.session_state['ma_mae']
        ma_mae_diff = st.session_state['ma_mae_diff']
        arima_mae = st.session_state['arima_mae']
        sarima_mae = st.session_state['sarima_mae']
        rf_mae = st.session_state['rf_mae']
        prop_mae = st.session_state['prop_mae']

        # Ensure that the MAE values are available
        try:
            model_names = ['Naive Forecast', 'Moving Average', 'Differenced Moving Average', 'ARIMA', 'SARIMA', 'Random Forest', 'Prophet']
            mae_values = [nf_mae, ma_mae, ma_mae_diff, arima_mae, sarima_mae, rf_mae, prop_mae]

            # Create a bar chart for MAE values
            plt.figure(figsize=(10, 6))
            plt.bar(model_names, mae_values, color='skyblue')
            plt.xlabel('Models', fontsize=12)
            plt.ylabel('Mean Absolute Error', fontsize=12)
            plt.title('Model Comparison: MAE Values', fontsize=14)
            plt.xticks(rotation=45, fontsize=10)
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(plt)

        except NameError as e:
            st.error(f"Error: {str(e)}. Please ensure all MAE values are calculated before comparing models.")

elif page == "Page 12: Forecasting with the Best Model":
    st.title("🤖 Forecasting with the Best Model")

    if "df_grouped" in st.session_state:
        df_grouped = st.session_state['df_grouped']

        # Button 1: Do Machine Learning
        if st.button("Forecast"):
            df_grouped.index = pd.to_datetime(df_grouped.index)

            forecast_steps = 10  # Number of steps to forecast
            forecast_index = pd.date_range(start=df_grouped.index[-1], periods=forecast_steps + 1, freq='D')[1:]

            fitted_sarima = st.session_state['fitted_sarima']
            forecast = fitted_sarima.forecast(steps=forecast_steps)

            forecast = pd.Series(forecast, index=forecast_index)

            plt.figure(figsize=(10, 6))
            plt.plot(df_grouped['New_Catches'], label='Historical Data', color='blue')
            plt.plot(forecast.index, forecast, label='Forecast', color='red')
            plt.legend()
            plt.title('SARIMA Forecast')
            st.pyplot(plt)


