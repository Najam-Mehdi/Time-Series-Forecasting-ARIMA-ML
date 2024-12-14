import pandas as pd
import urllib

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Load the provided CSV files to inspect their content
try:
    temperature_data = load_data("https://raw.githubusercontent.com/Najam-Mehdi/Insect-Prediction/refs/heads/main/docs/Temperature.csv")
    insect_data = load_data("https://raw.githubusercontent.com/Najam-Mehdi/Insect-Prediction/refs/heads/main/docs/Insect_Caught.csv")
except urllib.error.URLError as e:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    temperature_data = load_data("https://raw.githubusercontent.com/Najam-Mehdi/Insect-Prediction/refs/heads/main/docs/Temperature.csv")
    insect_data = load_data("https://raw.githubusercontent.com/Najam-Mehdi/Insect-Prediction/refs/heads/main/docs/Insect_Caught.csv")



# Display the first few rows and column information of each file
insect_info = insect_data.info(), insect_data.head()
temperature_info = temperature_data.info(), temperature_data.head()


# Convert Date_Time to datetime format
def date_parsing(df):
    df['Date'] = pd.to_datetime(df['Date_Time'], format='%d.%m.%Y %H:%M:%S').dt.date
    df.drop(columns=['Date_Time'], inplace=True)
    return df

insect_data = date_parsing(insect_data)
temperature_data = date_parsing(temperature_data)


# Remove duplicates in the temperature data (if any) by averaging duplicate timestamp values
temperature_data = temperature_data.groupby('Date').mean().reset_index()
insect_data = insect_data.groupby('Date').sum().reset_index()

# Merge datasets on Date_Time
merged_data = pd.merge(insect_data, temperature_data, on='Date', how='inner')

# Check the result of the merge and summary statistics
merged_data_info = merged_data.info(), merged_data.describe(), merged_data.head()


# export the merged data to a new CSV file
merged_data.to_csv("merged_data.csv", index=False)
print("Data exported to merged_data.csv")
