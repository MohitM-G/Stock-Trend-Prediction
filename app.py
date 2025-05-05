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
import datetime

# -------------------------
# 📚 Glossary Sidebar
# -------------------------
st.sidebar.title("📘 Glossary")
st.sidebar.markdown("""
### 📈 Moving Averages
- **100MA / 200MA**: Averages over 100 or 200 days to smooth trends.

### 🧠 Model Info
- **Trained Model**: A deep learning model trained on historical closing prices.
- **Input**: Last 100 days of closing data.
- **Output**: Predicted stock closing price.

### ⚙️ MinMaxScaler
- Scales data to range [0, 1].
- Boosts model training efficiency.

### 📊 Visualization
- Line graphs to compare actual vs predicted prices.
- Shows how well the model forecasts trends.

""")

st.title("""  
         📈 Stock Price Prediction
        This app uses a deep learning LSTM model to predict stock prices based on historical data.
        Built with: Python, Streamlit, Keras, Yahoo Finance API  
        Model Format: `.keras` (Keras v3+)

    """)

# -------------------------
# 📅 Date Picker
# -------------------------
st.sidebar.title(" Date Picker")
st.sidebar.markdown("### 📅 Date Range")
start = st.sidebar.date_input("Start Date", datetime.date(2010, 1, 1))
end = st.sidebar.date_input("End Date", datetime.date(2024, 12, 31))

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
try:
    df = yf.download(user_input, start, end)
    if df.empty:
        st.error(f"❌ No data found for '{user_input}'. Please check the ticker symbol.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error fetching data: {e}")
    st.stop()
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
st.subheader(f"Data from {start} to {end}")
st.write(df.describe())

# 🛡️ Handle empty data (e.g. due to Yahoo Finance rate limits)
if df.empty:
    st.error("❌ Failed to fetch data. Please check the ticker symbol or try again later.")
    st.stop()

# ─────────────────────────────────────
# 📆 Year Range Slider
# ─────────────────────────────────────
min_year = int(df.index.min().year)
max_year = int(df.index.max().year)

year_range = st.slider("Select Year Range", min_year, max_year, (2010, 2024))
filtered_df = df[(df.index.year >= year_range[0]) & (df.index.year <= year_range[1])]


# ─────────────────────────────────────
# 📉 Moving Average Chart
# ─────────────────────────────────────
st.subheader('📉 Closing Price vs Time Chart')
ma100 = filtered_df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100MA')
plt.plot(filtered_df.Close, label='Close')
plt.legend()
st.pyplot(fig)

st.subheader('📉 Closing Price vs Time Chart with 100MA & 200MA')
ma200 = filtered_df.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label='100MA')
plt.plot(ma200, 'g', label='200MA')
plt.plot(filtered_df.Close, 'b', label='Close')
plt.legend()
st.pyplot(fig2)

# ─────────────────────────────────────
# 🔄 Train/Test Split
# ─────────────────────────────────────
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):])

st.info(f"📦 Training Samples: {data_training.shape[0]} | Testing Samples: {data_testing.shape[0]}")

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# ─────────────────────────────────────
# 🤖 Load Model
# ─────────────────────────────────────
if not os.path.exists("keras_model.keras"):
    st.error("Model file 'keras_model.keras' not found. Please upload or download it.")
    st.stop()

model = load_model('keras_model.keras')

# ─────────────────────────────────────
# 📈 Prepare Test Data and Predict
# ─────────────────────────────────────
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scale_factor = 1/scaler.scale_[0]
y_predicted *= scale_factor
y_test *= scale_factor

# ─────────────────────────────────────
# 📊 Prediction Chart
# ─────────────────────────────────────
st.subheader('📉 Predictions vs Original')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# ─────────────────────────────────────
# 📤 Download Prediction Data
# ─────────────────────────────────────
pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_predicted.reshape(-1)})
csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Prediction Data", data=csv, file_name='predictions.csv', mime='text/csv')

# -------------------------
# 📧 Email Alert Section
# -------------------------
st.subheader("📧 Price Alert")
email = st.text_input("Enter your email for price alerts")
threshold = st.number_input("Alert if predicted price exceeds:", min_value=0.0)

if st.button("Set Alert"):
    if model and len(y_predicted) > 0 and max(y_predicted) > threshold:
        st.success(f"📬 Alert triggered! Price predicted to exceed {threshold}.")
    else:
        st.info("No alert triggered yet.")
