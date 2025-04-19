import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt
from src.utils import checkCacheOrTrain
from sklearn.metrics import r2_score

scaler = MinMaxScaler(feature_range=(0, 1))

@st.cache_resource
def implLSTM(filePath = "./data/air_quality.csv",testDataPath = "./data/test_data.csv", cachePath = "./Cache/LSTM"):
    """
    This function implements LSTM for time series forecasting. It preprocesses the data, trains the LSTM model,
    and then makes predictions based on the trained model. It visualizes the results in a Streamlit app.
    
    Args:
        filePath (str): Path to the dataset file (default is air_quality.csv).
        cachePath (str): Path to store cache files (default is ./Cache/LSTM).
    """
    st.title("LSTM for AQI Forecasting")

    # Displaying basic information about LSTM
    st.subheader("What is LSTM?")
    st.write("""
    LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) architecture, which is well-suited for time series forecasting tasks.
    It is capable of learning long-term dependencies in sequential data. LSTMs are widely used in various applications, such as predicting stock prices,
    weather forecasting, and in this case, forecasting the Air Quality Index (AQI).
    
    LSTM helps to address the vanishing gradient problem that traditional RNNs face by maintaining a cell state throughout the sequence processing.
    This allows LSTM networks to learn patterns over longer time horizons compared to standard RNNs.
    """)
    
    # Displaying dataset information
    st.subheader("Dataset Information")
    st.write("Dataset: AQI data for air quality prediction")

    scaledAirQuality = preprocessData(filePath)
    def create_sequences(data, time_step):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i])  # Include all features (AQI, PM, Temp, etc.)
            y.append(data[i, 0])  # Predict AQI (target)
        return np.array(X), np.array(y)

    # Create sequences with the updated feature set
    time_step = 30  # Using 30 previous days to predict the next day's AQI
    X, y = create_sequences(scaledAirQuality, time_step)

    # Reshape input for LSTM [samples, time_steps, features]
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Split the data into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    model = checkCacheOrTrain("LSTM",cachePath ,createLstmModel , X_train , y_train  ,X_test , y_test)
    predictAndPlot(model,X_train,X_test,y_test,testDataPath)



# Function to preprocess the dataset
def preprocessData(filePath):
    """
    Preprocess the data by reading it, handling missing values, and normalizing features.
    
    Args:
        filePath (str): Path to the dataset.
    
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    air_quality = pd.read_csv(filePath)  # Make sure this file is in the same directory
    air_quality['Date & Time'] = pd.to_datetime(air_quality['Date & Time'], format ='%m/%d/%y %I:%M %p', errors='coerce')
    st.subheader("Dataset Preview")
    st.write(air_quality.head())

    # Convert 'Date & Time' to datetime format if not already
    air_quality['Date & Time'] = pd.to_datetime(air_quality['Date & Time'])

    # Extract useful features from 'Date & Time'
    air_quality['Day of Week'] = air_quality['Date & Time'].dt.dayofweek  # Monday=0, Sunday=6
    air_quality['Month'] = air_quality['Date & Time'].dt.month  # 1 to 12
    air_quality['Hour'] = air_quality['Date & Time'].dt.hour  # 0 to 23
    air_quality['Day of Year'] = air_quality['Date & Time'].dt.dayofyear  # 1 to 365
    air_quality['Is Weekend'] = air_quality['Day of Week'].isin([5, 6]).astype(int)  # 1 for Saturday/Sunday, 0 for weekdays

    # Select relevant features including the extracted datetime-based features
    features = ['AQI', 'PM 1', 'PM 2.5', 'PM 10', 'Temp - C', 'Hum', 'Dew Point', 'Day of Week', 'Month', 'Hour', 'Is Weekend']

    # Scale all features (including temporal features) for LSTM
    air_quality_scaled = scaler.fit_transform(air_quality[features])
    return air_quality_scaled

# Function to create and train the LSTM model
def createLstmModel(X_train,y_train,X_test , y_test):
    """
    Creates and trains an LSTM model.
    
    Args:
        XTrain (np.array): Training features.
        yTrain (np.array): Training labels.
    
    Returns:
        model: Trained LSTM model.
    """
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    return lstm_model

# Function to predict using the LSTM model
def predictAndPlot(lstm_model,X_train , X_test , y_test ,testDataPath):
    """
    Makes predictions using the trained LSTM model.
    
    Args:
        model: The trained LSTM model.
        data (np.array): The data to make predictions on.
        timeStep (int): The time step used in training.
    
    Returns:
        np.array: Predictions from the LSTM model.
    """
    # Load test data
    test_data = pd.read_csv(testDataPath)  # Make sure this file is in the same directory
    test_data['Date & Time'] = pd.to_datetime(test_data['Date & Time'], format='%m/%d/%y %I:%M %p', errors='coerce')

    # Extract the date and ensure it's datetime
    daily_test_data = test_data.groupby(test_data['Date & Time'].dt.date)['AQI'].mean().reset_index()
    daily_test_data.columns = ['ds', 'y']

    # Convert 'ds' column in daily_test_data to datetime
    daily_test_data['ds'] = pd.to_datetime(daily_test_data['ds'])
    # Make predictions with LSTM
    y_pred_lstm = lstm_model.predict(X_test)

    # Inverse scale the predictions and actual values
    y_pred_lstm = scaler.inverse_transform(np.concatenate((y_pred_lstm, np.zeros((y_pred_lstm.shape[0], X_train.shape[2] - 1))), axis=1))[:, 0]
    y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X_train.shape[2] - 1))), axis=1))[:, 0]

    # Reshape the predictions and actual values to match the test data's daily granularity
    y_pred_lstm_daily = []
    y_test_actual_daily = []

    # Group the predictions by day (assuming the length of y_pred_lstm corresponds to the 15-minute data points)
    for i in range(len(daily_test_data)):
        # Get the indices of the predictions corresponding to this day
        start_idx = i * (len(y_pred_lstm) // len(daily_test_data))  # Starting index of the day
        end_idx = (i + 1) * (len(y_pred_lstm) // len(daily_test_data))  # Ending index of the day
        daily_pred = y_pred_lstm[start_idx:end_idx]  # Predictions for this day
        daily_actual = y_test_actual[start_idx:end_idx]  # Actual values for this day

        # Calculate the daily average
        y_pred_lstm_daily.append(np.mean(daily_pred))
        y_test_actual_daily.append(np.mean(daily_actual))

    # Convert the daily predictions and actuals to numpy arrays
    y_pred_lstm_daily = np.array(y_pred_lstm_daily)
    y_test_actual_daily = np.array(y_test_actual_daily)

    # Prepare the comparison DataFrame for LSTM
    comparison_lstm = pd.DataFrame({
        'ds': daily_test_data['ds'],
        'y_actual': y_test_actual_daily.flatten(),
        'y_predicted': y_pred_lstm_daily.flatten()
    })

    # Calculate residuals for LSTM
    comparison_lstm['residuals'] = comparison_lstm['y_actual'] - comparison_lstm['y_predicted']
    # Evaluation metrics
    mae = mean_absolute_error(comparison_lstm['y_actual'], comparison_lstm['y_predicted'])
    rmse = root_mean_squared_error(comparison_lstm['y_actual'], comparison_lstm['y_predicted'])
    r2 = r2_score(comparison_lstm['y_actual'], comparison_lstm['y_predicted'])
    accuracy = 100 - (mae / comparison_lstm['y_actual'].mean()) * 100

    # Display metrics
    st.title("Air Quality Index Prediction")
    st.markdown(f"##### LSTM Model - Mean Absolute Error: <span style='color:red;'>{mae:.2f}</span>", unsafe_allow_html=True)
    st.markdown(f"##### LSTM Model - Root Mean Square Error: <span style='color:red;'>{rmse:.2f}</span>", unsafe_allow_html=True)
    st.markdown(f"##### LSTM Model - R² Score (Explained Variance): <span style='color:green;'>{r2:.2%}</span>", unsafe_allow_html=True)
    st.markdown(f"##### LSTM Model - Estimated Accuracy: <span style='color:green;'>{accuracy:.2f}%</span>", unsafe_allow_html=True)


    # Plot actual vs predicted for LSTM
    fig2 = px.line(comparison_lstm, x='ds', y=['y_actual', 'y_predicted'], labels={'value': 'AQI', 'variable': 'Legend'}, title='LSTM: Actual vs Predicted AQI')
    st.plotly_chart(fig2)

    # Plot residuals for LSTM
    fig3 = px.scatter(comparison_lstm, x='ds', y='residuals', title='Residuals of LSTM Predictions')
    fig3.add_shape(type='line', x0=comparison_lstm['ds'].min(), y0=0, x1=comparison_lstm['ds'].max(), y1=0, line=dict(color='red', dash='dash'))
    st.plotly_chart(fig3)
