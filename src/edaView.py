# eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

# Function to load data
def loadData(filePath):
    """
    Loads data from the specified CSV file.
    
    Args:
        filePath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(filePath)
    df['Date & Time'] = pd.to_datetime(df['Date & Time'], format='%m/%d/%y %I:%M %p', errors='coerce')
    return df

# Function to clean data
def cleanData(df):
    """
    Cleans the dataset by handling missing values and other potential issues.
    
    Args:
        df (pd.DataFrame): Dataframe to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Example: Drop rows with missing values (customize as needed)
    df_cleaned = df.dropna()
    
    # You can also handle other cleaning tasks like removing duplicates, etc.
    df_cleaned = df_cleaned.drop_duplicates()
    
    return df_cleaned

# Function to get summary statistics of the dataset
def getSummaryStats(df):
    """
    Generates summary statistics for numerical columns.
    
    Args:
        df (pd.DataFrame): Dataframe to generate statistics for.
    
    Returns:
        pd.DataFrame: Summary statistics.
    """
    return df.describe()

# Function to visualize the distribution of a given column
def plotColumnDistribution(df, column):
    """
    Plots the distribution of a specified column.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data.
        column (str): Name of the column to plot.
    
    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    hist = sns.histplot(df[column], kde=True, color='#1E90FF', bins=30)
    kde_line = hist.lines[0]  # Access the KDE line
    kde_line.set_color('#FF6347') 
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Function to visualize correlation matrix
def plotCorrelationMatrix(df):
    """
    Plots a heatmap of the correlation matrix.
    
    Args:
        df (pd.DataFrame): Dataframe to compute correlation.
    
    Returns:
        None
    """
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    st.pyplot(plt)

# Function to visualize pairwise relationships between numerical columns
def plotPairwiseRelations(df):
    """
    Plots pairwise relationships between numerical columns in the dataset.
    
    Args:
        df (pd.DataFrame): Dataframe to plot pairwise relationships.
    
    Returns:
        None
    """
    sns.pairplot(df)
    st.pyplot(plt)

# Function to visualize missing data
def plotMissingData(df):
    """
    Plots a heatmap of missing values.
    
    Args:
        df (pd.DataFrame): Dataframe to check for missing values.
    
    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    st.pyplot(plt)

# Function to visualize the distribution of a categorical column
def plotCategoricalDistribution(df, column):
    """
    Plots the distribution of a categorical column.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data.
        column (str): Name of the categorical column to plot.
    
    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=df, palette='Set2')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    st.pyplot(plt)

# Main function for Streamlit app
def runEDA(filePath , htmlFile = "./data/aqi_book.html"):
    st.title("Air Quality Index (AQI) Explanation")

    st.subheader("What is AQI?")
    st.write("""
        The **Air Quality Index (AQI)** is a numerical scale used to measure and communicate the level of air pollution in the atmosphere. It is based on the concentration of several pollutants in the air, including:
        - PM2.5
        - PM10
        - Nitrogen Dioxide (NO₂)
        - Ozone (O₃)
        - Sulfur Dioxide (SO₂)
        - Carbon Monoxide (CO)

        The AQI value helps to determine the quality of the air, and based on the value, individuals can gauge whether it is safe to go outside, especially for vulnerable populations.
    """)

    st.subheader("AQI Categories")
    st.write("""
        The AQI categories are as follows:
        - **0–50 (Good)**: Air quality is considered satisfactory, and air pollution poses little or no risk.
        - **51–100 (Moderate)**: Air quality is acceptable; however, some pollutants may be a concern for a few people.
        - **101–150 (Unhealthy for Sensitive Groups)**: Sensitive groups may experience health effects.
        - **151–200 (Unhealthy)**: Everyone may begin to experience health effects.
        - **201–300 (Very Unhealthy)**: Health alert: everyone may experience more serious health effects.
        - **301–500 (Hazardous)**: Health warning of emergency conditions: the entire population is likely to be affected.
    """)
    # Load the dataset
    df = loadData(filePath)
    
    # Clean the dataset
    df_cleaned = cleanData(df)
    
    # Display basic info about the dataset
    st.title("Exploratory Data Analysis (EDA)")
    st.write("### Dataset Overview")
    st.write(df_cleaned.head())
    st.write("### Summary Statistics")
    st.write(getSummaryStats(df_cleaned))
    
    # Visualizations
    st.subheader("Distribution of Columns")
    column = st.selectbox('Select column for distribution plot', df_cleaned.columns)
    plotColumnDistribution(df_cleaned, column)
    
    st.subheader("Correlation Matrix")
    plotCorrelationMatrix(df_cleaned)
    
    # st.subheader("Pairwise Relationships")
    # plotPairwiseRelations(df_cleaned)
    
    # with open(htmlFile, 'r') as file:
    #     html_content = file.read()
    # st.subheader("Profiler Result:")
    # st.components.v1.html(html_content, height=600)