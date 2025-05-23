# fbprophet.py
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import streamlit as st
from src.utils import checkCacheOrTrain

@st.cache_resource
def implProphet(filePath = "./data/air_quality.csv", cachePath = "./Cache/Prophet"):
    # Display basic information about FBProphet
    st.title("FB Prophet Model: Air Quality Prediction")
    
    st.write("""
        **FBProphet** is a forecasting tool designed to handle time series data. It is robust to missing data, shifts in the trend, and large outliers. 
        It works well with daily or seasonal data and is particularly suited for forecasting tasks where seasonal patterns and holiday effects are important.
        
        FBProphet provides the following key features:
        - **Handles missing data**: It automatically handles gaps in the time series.
        - **Flexible seasonality modeling**: It can model yearly, weekly, and daily seasonality.
        - **Holiday effects**: It allows for the inclusion of holidays to adjust the forecasts.
        - **Robustness to outliers**: It is designed to handle sudden spikes or drops in the data.
    """)
    
    # Show the dataset preview
    data = preprocessData(filePath)
    st.subheader("Dataset Preview")
    st.write(data.head())
    
    model = checkCacheOrTrain("FBProphet",cachePath,trainProphet,data)
    forecast = forecastProphet(model)
    plotForecast(model , forecast, data)

    

# Function to preprocess the dataset for FBProphet
def preprocessData(filePath):
    """
    Preprocesses the data for FBProphet.
    
    Args:
        filePath (str): Path to the dataset.
    
    Returns:
        pd.DataFrame: Preprocessed data with columns 'ds' and 'y' as required by Prophet.
    """
    air_quality = pd.read_csv(filePath)
    air_quality['Date & Time'] = pd.to_datetime(air_quality['Date & Time'], format='%m/%d/%y %I:%M %p', errors='coerce')
    
    # Prepare the data for Prophet
    data = pd.DataFrame()
    data['ds'] = air_quality['Date & Time']
    data['y'] = air_quality['AQI']
    
    return data

# Function to train the FBProphet model
def trainProphet(df):
    """
    Trains the FBProphet model on the preprocessed data.
    
    Args:
        df (pd.DataFrame): The preprocessed data with 'ds' and 'y' columns.
    
    Returns:
        model: Trained FBProphet model.
    """
    model = Prophet()
    model.fit(df)
    return model

# Function to make future predictions with the trained model
def forecastProphet(model, periods=30):
    """
    Makes future predictions using the trained FBProphet model.
    
    Args:
        model: Trained FBProphet model.
        periods (int): Number of days to forecast into the future.
    
    Returns:
        pd.DataFrame: Predictions with 'ds' and 'yhat' columns.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Function to plot the forecasted values
def plotForecast(model , forecast, data):
    """
    Plots the forecasted values with actual values and residuals.
    
    Args:
        forecast (pd.DataFrame): The forecasted values from the Prophet model.
        data (pd.DataFrame): The actual data used for training.
    
    Returns:
        None
    """
    # Merge the forecast with actual values for comparison
    comparison = pd.merge(forecast[['ds', 'yhat']], data, on='ds')
    comparison.rename(columns={'yhat': 'y_predicted', 'y': 'y_actual'}, inplace=True)

    # Check for missing or invalid data
    st.write("Comparison Dataframe after merging: ", comparison.head())
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(comparison['y_actual'], comparison['y_predicted'])
    rmse = root_mean_squared_error(comparison['y_actual'], comparison['y_predicted'])

    # Streamlit layout for Prophet
    st.title("Air Quality Index Prediction")
    st.markdown(f"##### Fb Prophet Model - Mean Absolute Error: <span style='color:red;'>{mae}</span>", unsafe_allow_html=True)
    st.markdown(f"##### Fb prophet Model - Root Mean Square Error: <span style='color:red;'>{rmse}</span>", unsafe_allow_html=True)
    color_map = {
    'y_actual': '#1E90FF',  # Change 'blue' to your desired color for actual values
    'y_predicted': '#FF6347'  # Change 'red' to your desired color for predicted values
    }
    from sklearn.metrics import r2_score

    # Calculate additional metrics
    r2 = r2_score(comparison['y_actual'], comparison['y_predicted'])
    accuracy = 100 - (mae / comparison['y_actual'].mean()) * 100

    # Display metrics in Streamlit
    st.markdown(f"##### FB Prophet Model - R² Score (Explained Variance): <span style='color:green;'>{r2:.2%}</span>", unsafe_allow_html=True)
    st.markdown(f"##### FB Prophet Model - Estimated Accuracy: <span style='color:green;'>{accuracy:.2f}%</span>", unsafe_allow_html=True)


    # Plot actual vs predicted for Prophet
    fig1 = px.line(comparison, x='ds', y=['y_actual', 'y_predicted'], labels={'value': 'AQI', 'variable': 'Legend'}, title='Prophet: Actual vs Predicted AQI', color_discrete_map=color_map)
    st.plotly_chart(fig1)

        # Filter the forecast for the next 30 days
    forecast_30_days = forecast[forecast['ds'] > data['ds'].max()]

    # Plot predicted AQI for the next 30 days (Plotly)
    fig2 = px.line(forecast_30_days, x='ds', y='yhat', labels={'yhat': 'Predicted AQI'}, title='FBProphet: Predicted AQI for the Next 30 Days', color_discrete_map={'yhat': '#FF6347'})
    st.plotly_chart(fig2)

    # Plot residuals for Prophet
    comparison['residuals'] = comparison['y_actual'] - comparison['y_predicted']
    fig3 = px.scatter(comparison, x='ds', y='residuals', title='Residuals of Prophet Predictions' , color_discrete_sequence=['#1E90FF'])
    fig3.add_shape(type='line', x0=comparison['ds'].min(), y0=0, x1=comparison['ds'].max(), y1=0, line=dict(color='#FF6347', dash='dash'))
    st.plotly_chart(fig3)

    # Additional plot: AQI over time (original data)
    fig4 = px.line(data, x='ds', y='y', title='Air Quality Index Over Time',color_discrete_sequence=['#1E90FF'])
    st.plotly_chart(fig4)

    # Additional plot: Histogram of AQI values
    fig5 = px.histogram(data, x='y', title='Distribution of AQI Values' , color_discrete_sequence=['#1E90FF'])
    st.plotly_chart(fig5)

    # Plot Prophet components
    st.subheader("Prophet Model Components")
    components_fig = model.plot_components(forecast)
    st.pyplot(components_fig)

    # plt.figure(figsize=(20, 6))
    # plt.plot(comparison['ds'], comparison['y_actual'], label='Actual AQI', marker='o', color='#1E90FF')
    # plt.plot(comparison['ds'], comparison['y_predicted'], label='Predicted AQI', marker='x', color='#FF6347')
    # plt.title('Actual vs Predicted AQI (Matplotlib)', fontsize=16)
    # plt.xlabel('Date')
    # plt.ylabel('AQI')
    # plt.legend()
    # st.pyplot(plt)

     # Add a Matplotlib plot for Predicted AQI (Next 30 Days)
    plt.figure(figsize=(20, 6))
    plt.plot(forecast_30_days['ds'], forecast_30_days['yhat'], label='Predicted AQI (Next 30 Days)', marker='x', color='#FF6347')
    plt.title('Predicted AQI for the Next 30 Days (Matplotlib)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.legend()
    st.pyplot(plt)

    # Add a Matplotlib plot for both Actual AQI (historical) and Predicted AQI (Next 30 Days)
    plt.figure(figsize=(20, 6))

    # Plot Actual AQI (Historical data)
    plt.plot(comparison['ds'], comparison['y_actual'], label='Actual AQI', marker='o', color='#1E90FF')

    # Plot Predicted AQI (Next 30 Days)
    plt.plot(forecast_30_days['ds'], forecast_30_days['yhat'], label='Predicted AQI (Next 30 Days)', marker='x', color='#FF6347')

    # Adding title and labels
    plt.title('Actual vs Predicted AQI (Next 30 Days)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('AQI')

    # Adding a legend to differentiate the actual vs predicted values
    plt.legend()

    # Display the plot
    st.pyplot(plt)
