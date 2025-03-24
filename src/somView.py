import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
import umap
from src.utils import checkCacheOrTrain

# view to render on screen!
@st.cache_resource
def implSOM(filePath = "data/march.csv", cachePath = "./Cache/SOM" , epochs = 20):
    # check cache or train model
    som = checkCacheOrTrain("SOM",cachePath,train , filePath , epochs)

    # Display basic information about SOM
    st.title("Self-Organizing Map (SOM) Model")
    
    st.write("""
        The **Self-Organizing Map (SOM)** is a type of unsupervised learning algorithm 
        that uses neural networks to create a low-dimensional (usually 2D) grid to represent 
        high-dimensional data. SOM is particularly useful for clustering and visualizing 
        complex data in a way that preserves topological relationships between data points.
    """)
    
    st.write("### Key Points of SOM:")
    st.write("""
        - Unsupervised learning
        - Clustering and data visualization
        - Creates a 2D map of high-dimensional data
        - Topological relationships preserved
    """)

     # Show the dataset preview
    data = readData(filePath)
    st.subheader("Dataset Preview")
    st.write(data.head())
    
    # Show SOM visualizations
    st.subheader("SOM Visualizations")
    plot_som_grid(som)  # Function to generate SOM grid visualization
    # plot_som_distance_map(som)  # Function to generate SOM distance map


def getFeatArray(df, feature,seperation):
    # Group data by date and collect feature values into lists
    feature_vectors = df.groupby(df['dateTime'].dt.date)[feature].apply(list).values

    # Normalize the feature vectors using MinMaxScaler
    scaler = MinMaxScaler()
    normalized_vectors = []
    for vec in feature_vectors:
        # Reshape the vector for the scaler (needs a 2D array)
        vec_2d = np.array(vec).reshape(-1, 1)
        normalized_vec = scaler.fit_transform(vec_2d)
        normalized_vec = normalized_vec.flatten() # Flatten back to 1D
        for (index,entry) in enumerate(normalized_vec):
            normalized_vec[index] = entry + seperation
        normalized_vectors.append(normalized_vec)

    return normalized_vectors

def readData(filePath):
    df = pd.read_csv(filePath,encoding="latin-1",skiprows=5)
    columns_to_keep = ['Date & Time','AQI', 'Temp - °C', 'PM 10 - ug/m³', 'PM 2.5 - ug/m³', 'High Heat Index - °C']
    df = df[columns_to_keep]
    df = df.rename(columns={
        'Date & Time': 'dateTime',
        'AQI': 'aqi',
        'Temp - °C': 'temp',
        'PM 10 - ug/m³': 'pM10',
        'PM 2.5 - ug/m³': 'pM25',
        'High Heat Index - °C': 'highHeatIndex'
    })
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    df = df[df['dateTime'].dt.month != 4] # drop april
    return df


def train(filePath , epochs):

    df = readData(filePath)

    temp = getFeatArray(df, 'temp',0)
    aqi = getFeatArray(df,"aqi",1)
    pM10 = getFeatArray(df,"pM10",2)
    pM25 = getFeatArray(df,"pM25",3)
    highHeatIndex = getFeatArray(df,"highHeatIndex",4)
    som = MiniSom(x=100, y=100, input_len=96, sigma=1.0, learning_rate=0.1)
    print("Check for Cache , Training locally not recommended.")

    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        som.train(temp, 1000, verbose=True)
        som.train(aqi, 1000, verbose=True)
        som.train(pM10, 1000, verbose=True)
        som.train(pM25, 1000, verbose=True)
        som.train(highHeatIndex, 1000, verbose=True)
    return som

def plot_som_grid(som):
    distance_map = som.distance_map()
    plt.figure(figsize=(10, 10))
    plt.pcolor(distance_map.T, cmap='bone_r')  # plotting the distance map as background
    plt.colorbar()
    plt.title('SOM Distance Map')
    plt.xlabel('x')
    plt.ylabel('y')
    st.pyplot(plt)

# def plot_som_distance_map(som):
#     # Get the winning neuron for each data point
#     winners = [som.winner(x) for x in np.array(temp+aqi+pM10+pM25+highHeatIndex)]

#     # Create a grid to represent the SOM
#     grid = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))

#     # Count the number of data points assigned to each neuron
#     for winner in winners:
#     grid[winner[0], winner[1]] += 1

#     # Visualize the SOM grid
#     plt.figure(figsize=(10, 10))
#     plt.pcolor(grid, cmap='viridis')  # Use 'viridis' or other colormaps
#     plt.colorbar()
#     plt.title('SOM Grid Visualization')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()