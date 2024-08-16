import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from streamlit_tags import st_tags

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Call Analysis',
    page_icon=':earth_americas:',  # This is an emoji shortcode. Could be a URL too.
)

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



def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Check for a key that is almost always present in valid tickers
        if info.get('symbol') is not None:
            return True
    except Exception as e:
        return False
    return False

# Allow user to input any ticker symbol
tickers = st_tags(
    value = ["MSFT"],
    label="Enter one stock ticker at a time."
)

# Process the user input into a list of tickers

# Validate each ticker
selected_stocks = []
for ticker in tickers:
    if is_valid_ticker(ticker):
        selected_stocks.append(ticker)
    else:
        st.warning(f"{ticker} is not a valid ticker symbol.")



def get_call_price(stock):
    last = yf.Ticker(stock)
    history_data = last.history(period="1d")
    
    if history_data.empty:
        st.warning(f"No data available for {stock}.")
        return None

    S = history_data['Close'][0]

    sigma = calculate_volatility(stock_data_df[stock_data_df['Stock Ticker'] == stock]['Close'])

    d1 = (np.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    C = S*Nd1-K*np.exp(-R*T)*Nd2

    return C

def get_put_price(stock):
    last = yf.Ticker(stock)
    history_data = last.history(period="1d")
    
    if history_data.empty:
        st.warning(f"No data available for {stock}.")
        return None

    S = history_data['Close'][0]

    sigma = calculate_volatility(stock_data_df[stock_data_df['Stock Ticker'] == stock]['Close'])

    d1 = (np.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    Nd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(-d2)

    P = -S*Nd1+K*np.exp(-R*T)*Nd2

    return P

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

    st.header('Stock Price Over Time', divider='gray')
    old_data_df = pd.DataFrame()
    for stock in selected_stocks:
        stock_data = yf.download(stock, period='max')
        if not stock_data.empty:
            first_trading_date = stock_data.index.min()
            first_date_str = first_trading_date.strftime('%Y-%m-%d')
            old_df = get_stock_data(ticker=stock, start_date=first_date_str, end_date=end_date_str)
            old_df['Stock Ticker'] = stock
            old_data_df = pd.concat([old_data_df, old_df])

    if not stock_data_df.empty:
        st.line_chart(
            old_data_df,
            x='Date',  # Use 'Date' to get daily prices
            y='Close',  # Assuming we want to plot the closing price
            color='Stock Ticker',
        )
    else:
        st.warning("No data available to display.")

    st.header('Stock Volatility', divider='gray')
    for stock in selected_stocks:
        stock_prices = stock_data_df[stock_data_df['Stock Ticker'] == stock]['Close']
        if not stock_prices.empty:
            volatility = calculate_volatility(stock_prices)
            st.write(f"Volatility for {stock}: {volatility:.4f}")
    
    

    st.header('Black-Scholes Suggested European Call Price', divider='gray')
            
    R = st.number_input("Interest Rate", min_value=-0.01, max_value=1.0, value = 0.05)
    K = st.number_input("Strike Price", min_value=0.0, value = 100.0, step=10.0)
    T = st.number_input("Time to maturity (years)", min_value=0.0, value = 1.0)


    for stock in selected_stocks:
        st.text("")
        st.text("")
        st.text("")
        last = yf.Ticker(stock)
        history_data = last.history(period="1d")
    
        if history_data.empty:
            st.warning(f"No data available for {stock}.")
        else:    
            final = history_data['Close'][0]
            st.write(f"Current price of {stock}: {final:.2f}")
            Call_price = get_call_price(stock)
            st.write(f"Call price for {stock}: {Call_price:.2f}")
            Put_price = get_put_price(stock)
            st.write(f"Put price for {stock}: {Put_price:.2f}")


    
        
else:
    st.warning("Please select at least one stock.")
