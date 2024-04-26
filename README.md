# 📈📉💲 Stock-Trend-Prediction


![1stock](https://github.com/MohitM-G/Stock-Trend-Prediction/assets/147160445/0786ee1a-ec73-434e-87a4-b7482b5858d7)


![predicted](https://github.com/MohitM-G/Stock-Trend-Prediction/assets/147160445/0e9d575e-b5fd-416e-84da-b8c3a6b686bd)


This project focuses on predicting the trends of stock prices using machine learning techniques. It analyzes historical stock data along with various features to forecast whether the stock price will increase, decrease, or remain stable in the near future.

## ℹ Introduction

Forecasting stock market trends is inherently challenging due to its complexity and volatility. This project utilizes machine learning algorithms to analyze historical stock data and provide predictive insights. The goal is to aid investors in making informed decisions based on predictive analytics.

## 📝 Features

- **Data Collection**: Retrieves historical stock data from various sources such as Yahoo Finance, Alpha Vantage, etc.
- **Feature Engineering**: Extracts relevant features from the stock data, including technical indicators, sentiment analysis of news articles, market sentiment, etc.
- **Model Training**: Utilizes machine learning models such as Random Forest, Gradient Boosting, LSTM, etc., to train predictive models.
- **Evaluation**: Assesses model performance using metrics like accuracy, precision, recall, and F1-score.
- **Prediction**: Generates predictions for future stock trends based on the trained models.
- **Visualization**: Visualizes historical data, model predictions, and evaluation metrics using graphs and charts.

## 🖥 Installation
🛠 Requirements
▸ Python 3.5+
▸ Visual Studio Code (VS Code) 2017
▸ Modules = numpy, pandas, scikit-learn, 
             matplotlib, tensorflow, keras, nltk

# ⚙ Setup
1. Clone the repository:

    ```bash
    git clone https://github.com/MohitM-G/stock-trend-prediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd stock-trend-prediction
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the data collection script to retrieve historical stock data:

    ```bash
    python data_collection.py
    ```

2. Preprocess the data and extract features:

    ```bash
    python feature_engineering.py
    ```

3. Train the predictive models:

    ```bash
    python train_models.py
    ```

4. Evaluate model performance:

    ```bash
    python evaluate_models.py
    ```

5. Make predictions:

    ```bash
    python predict.py
    ```
