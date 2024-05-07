import streamlit as st

import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"
import yfinance as yf

from datetime import datetime, timedelta
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import numpy as np

def fetch_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def fit_arima_model(data):
    close_prices = data['Close']

    # Menentukan parameter model ARIMA (p, d, q)
    order = (5, 1, 1)  # Contoh parameter

    # Melatih model ARIMA
    model = ARIMA(close_prices, order=order)
    fitted_model = model.fit()

    return fitted_model

def fit_linear_regression(data):
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data['Close'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    return model

def main():
    st.title('Streamlit Saham')

    symbol = st.sidebar.text_input("Masukkan Simbol Saham (contoh: AAPL untuk Apple Inc.)", value='AAPL')
    start_date = st.sidebar.date_input("Tanggal Awal", value=(datetime.now() - timedelta(days=365)))
    end_date = st.sidebar.date_input("Tanggal Akhir", value=datetime.now())

    stock_data = fetch_stock_data(symbol, start_date, end_date)

    if stock_data is not None:
        st.write(f"Data Saham {symbol}")
        st.write(stock_data)

        # Membuat grafik candlestick
        fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                             open=stock_data['Open'],
                                             high=stock_data['High'],
                                             low=stock_data['Low'],
                                             close=stock_data['Close'])])
        
        fig.update_layout(title=f'Grafik Candlestick untuk {symbol}',
                          xaxis_title='Tanggal',
                          yaxis_title='Harga',
                          xaxis_rangeslider_visible=False)
        
        st.plotly_chart(fig)

        # Melatih model ARIMA
        model_arima = fit_arima_model(stock_data)

        # Membuat prediksi untuk 30 hari ke depan
        forecast = model_arima.forecast(steps=7)  # Contoh: prediksi untuk 30 hari
        forecast_dates = [stock_data.index[-1] + timedelta(days=i) for i in range(1, 31)]

        # Menambahkan prediksi ke dalam grafik
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Prediksi ARIMA'))

        # Melatih model regresi linear
        model_lr = fit_linear_regression(stock_data)

        # Menambahkan garis tren regresi linear di bawah grafik candlestick
        x_values = np.array(range(len(stock_data)))
        y_values = model_lr.predict(x_values.reshape(-1, 1))
        fig.add_trace(go.Scatter(x=stock_data.index, y=y_values.flatten(), mode='lines', name='Tren Regresi Linear', line=dict(color='orange', width=2, dash='dash')))

        # Menentukan tren pasar (BULLISH atau BEARISH) berdasarkan koefisien regresi
        trend_text = "BULLISH" if model_lr.coef_[0][0] > 0 else "BEARISH"

        # Menambahkan teks tren pasar ke dalam grafik
        fig.add_annotation(x=stock_data.index[-1], y=y_values[-1][0], text=f"Tren Pasar: {trend_text}", showarrow=True, arrowhead=1)

        st.plotly_chart(fig)

if __name__ == "__main__":
    main()


