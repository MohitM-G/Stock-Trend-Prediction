# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras import load_model
import streamlit as st

st.title("Stock Trend Prediction")


start = '2000-01-01'
end = '2023-12-31'

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start, end)


# Describing Data
st.subheader('Data from 2000 - 2023')
st.write(df.describe())
