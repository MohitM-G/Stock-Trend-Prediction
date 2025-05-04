import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import os
import urllib.request
from keras.models import load_model
import plotly.graph_objects as go

st.title("Stock Price Prediction")
# ─────────────────────────────────────
# ℹ️ About Section
# ─────────────────────────────────────
with st.expander("ℹ️ About This App"):
    st.write("""
        This app uses a deep learning LSTM model to predict stock prices based on historical data.
        Built with: Python, Streamlit, Keras, Yahoo Finance API  
        Model Format: `.keras` (Keras v3+)

    """)

start = '2000-01-01'
end = '2024-12-31'

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start, end)
filtered_df = df.copy()

# 💰 Show Latest Price
try:
    ticker_info = yf.Ticker(user_input)
    latest_price = ticker_info.history(period="1d")['Close'].iloc[-1]
    
    # Detect currency (default to USD if unknown)
    currency = ticker_info.info.get("currency", "USD")
    
    if currency == "INR":
        symbol = "₹"
    else:
        symbol = "$"
    
    st.metric(label=f"Latest Closing Price of {user_input}", value=f"{symbol}{latest_price:.2f}")
except Exception as e:
    st.warning(f"Unable to fetch latest price. ({str(e)})")

# ─────────────────────────────────────
# 🧾 Raw Dataset
# ─────────────────────────────────────
show_raw = st.checkbox("📂 Show Raw Data")

if show_raw:
    st.subheader('📊 Raw Dataset')
    st.write(df)

# Describing Data
st.subheader('Data from 2000 - 2024')
st.write(df.describe())

# ─────────────────────────────────────
# 📆 Year Range Slider
# ─────────────────────────────────────
min_year = int(df.index.min().year)
max_year = int(df.index.max().year)

year_range = st.slider("Select Year Range", min_year, max_year, (2010, 2024))
filtered_df = df[(df.index.year >= year_range[0]) & (df.index.year <= year_range[1])]


# Visualizations
st.subheader('Closing Price vs Time Chart')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()  # 100 day moving average 
ma200 = df.Close.rolling(200).mean()  # 200 day moving average
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig) #



# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

print(data_training.shape)
print(data_testing.shape)


# Scaling Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


data_training_array = scaler.fit_transform(data_training)


# Load My Model
model = load_model('keras_model.keras', compile=False)


# Testing Part
past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making Predictions
y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final Graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# ─────────────────────────────────────
# 📤 Download Prediction Data
# ─────────────────────────────────────
pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_predicted.reshape(-1)})
csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Prediction Data", data=csv, file_name='predictions.csv', mime='text/csv')
