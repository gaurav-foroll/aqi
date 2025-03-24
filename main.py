import streamlit as st

# Import the functions from the src folder
from src.edaView import runEDA
from src.prophetView import implProphet
from src.lstmView import implLSTM
from src.somView import implSOM

def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("PAGES:", options=["EDA", "FBProphet", "LSTM", "SOM"])

    # Switch-like behavior with if-elif
    if page == "EDA":
        runEDA(filePath="./data/air_quality.csv")
    elif page == "FBProphet":
        implProphet()
    elif page == "LSTM":
        implLSTM()
    elif page == "SOM":
        implSOM(filePath="./data/march.csv")

if __name__ == "__main__":
    main()
