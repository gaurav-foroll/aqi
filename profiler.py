from ydata_profiling import ProfileReport
import pandas as pd

df =pd.read_csv("./data/air_quality.csv")

profile= ProfileReport(df, title="AQI Books")

profile.to_notebook_iframe()
profile.to_file("./data/aqi_book.html")