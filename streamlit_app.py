import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Call Analysis',
    page_icon=':earth_americas:',  # This is an emoji shortcode. Could be a URL too.
)

available_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# -----------------------------------------------------------------------------
# Declare some useful functions.

st.cache_data.clear()
@st.cache_data
def get_stock_data(ticker, start_date='2020-01-01', end_date='2023-12-31'):
    try:
        # Suppress progress bar
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Ensure the index is of type DatetimeIndex
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data.index = pd.to_datetime(stock_data.index)
        
        # Extract year and month
        stock_data['Year'] = stock_data.index.year
        stock_data['Month'] = stock_data.index.month
        stock_data['Date'] = stock_data.index  # Ensure 'Date' is part of the DataFrame
        
        return stock_data
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# -----------------------------------------------------------------------------
# Draw the actual page

end_date = datetime.today()
start_date = end_date - timedelta(days=6*30)  # Approximate 6 months as 180 days
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

selected_stocks = st.multiselect(
    'Which stocks do you want to view?',
    options=available_stocks,
    default=['MSFT']
)

def get_call_price(stock):
    last = yf.Ticker(stock)
    S = last.history(period="1d")['Close'][0]
    K = 95
    R = 0.05
    T = 1
    sigma = calculate_volatility(stock_data_df[stock_data_df['Stock Ticker'] == stock]['Close'])

    d1 = (np.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    C = S*Nd1-K*np.exp(-R*T)*Nd2

    return C



stock_data_df = pd.DataFrame()
if selected_stocks:
    # Filter the data
    stock_data_df = pd.DataFrame()
    for stock in selected_stocks:
        stock_df = get_stock_data(ticker=stock, start_date=start_date_str, end_date=end_date_str)
        stock_df['Stock Ticker'] = stock
        stock_data_df = pd.concat([stock_data_df, stock_df])

    def calculate_volatility(prices):
        if len(prices) < 2:
            return np.nan  # Not enough data to calculate volatility
        returns = prices.pct_change().dropna()
        # Calculate the standard deviation of returns
        volatility = np.std(returns) * np.sqrt(252)
        return volatility

    st.header('Stock Volatility', divider='gray')
    for stock in selected_stocks:
        stock_prices = stock_data_df[stock_data_df['Stock Ticker'] == stock]['Close']
        if not stock_prices.empty:
            volatility = calculate_volatility(stock_prices)
            st.write(f"Volatility for {stock}: {volatility:.4f}")

    st.header('Stock price over time', divider='gray')
    if not stock_data_df.empty:
        st.line_chart(
            stock_data_df,
            x='Date',  # Use 'Date' to get daily prices
            y='Close',  # Assuming we want to plot the closing price
            color='Stock Ticker',
        )
    else:
        st.warning("No data available to display.")

    st.header('Call Price', divider='gray')
    
    for stock in selected_stocks:
        Call_price = get_call_price(stock)
        st.write(f"Call price for {stock}: {Call_price:.4f}")
        
else:
    st.warning("Please select at least one stock.")
