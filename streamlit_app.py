import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from scipy.stats import norm
from streamlit_tags import st_tags
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.pyplot as plt



st.set_page_config(
    page_title='Portfolio Optimization',
    page_icon=':earth_americas:',  
)

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Portfolio Optimization"])

if page == "Call-Analysis":
    st.cache_data.clear()
    @st.cache_data
    def get_stock_data(ticker, start_date='2020-01-01', end_date='2023-12-31'):
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not isinstance(stock_data.index, pd.DatetimeIndex):
                stock_data.index = pd.to_datetime(stock_data.index)
            
            stock_data['Year'] = stock_data.index.year
            stock_data['Month'] = stock_data.index.month
            stock_data['Date'] = stock_data.index  
            
            return stock_data
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return pd.DataFrame() 


    end_date = datetime.today()
    start_date = end_date - timedelta(days=6*30)  
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')



    def is_valid_ticker(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if info.get('symbol') is not None:
                return True
        except Exception as e:
            return False
        return False

    tickers = st_tags(
        value = ["MSFT"],
        label="Enter one stock ticker at a time."
    )

    selected_stocks = []
    for ticker in tickers:
        if is_valid_ticker(ticker):
            selected_stocks.append(ticker)
        else:
            st.warning(f"{ticker} is not a valid ticker symbol.")



    def get_call_price(stock):
        last = yf.Ticker(stock)
        history_data = last.history(period="5d")
        if history_data.empty:
            st.warning(f"No data available for {stock}.")
        else:
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
        history_data = last.history(period="5d")
        
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
        stock_data_df = pd.DataFrame()
        for stock in selected_stocks:
            stock_df = get_stock_data(ticker=stock, start_date=start_date_str, end_date=end_date_str)
            stock_df['Stock Ticker'] = stock
            stock_data_df = pd.concat([stock_data_df, stock_df])

        def calculate_volatility(prices):
            if len(prices) < 2:
                return np.nan  
            returns = prices.pct_change().dropna()
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
            old_data_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in old_data_df.columns]

            if 'Date' in old_data_df.columns:
                if old_data_df['Date'].dtype != 'datetime64[ns]':
                    old_data_df['Date'] = pd.to_datetime(old_data_df['Date'])

            pivot_df = old_data_df.pivot(index='Date', columns='Stock Ticker', values='Close')

            st.line_chart(pivot_df)
            
        else:
            st.warning("No data available to display.")

        
        

        st.header('Black-Scholes Suggested European Call And Put Price', divider='gray')
                
        R = st.number_input("Interest Rate", min_value=-0.01, max_value=1.0, value = 0.05)
        K = st.number_input("Strike Price", min_value=0.0, value = 100.0, step=10.0)
        T = st.number_input("Time to maturity (years)", min_value=0.0, value = 1.0)

        css = """
            <style>
            .container {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                align-items: flex-start;
                gap: 20px;
                margin: 0 auto;
                width: 100%;
                box-sizing: border-box;
            }

            .circle {
                width: 200px;
                height: 200px;
                background-color: #3498db;
                border-radius: 50%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                font-size: 16px;
                color: white;
                text-align: center;
                padding: 10px;
                box-sizing: border-box;
            }

            .circle .content {
                font-size: 24px;
                font-weight: bold;
            }
            </style>
        """
        html_content = '<div class="container">'

        for stock in selected_stocks:
            last = yf.Ticker(stock)
            history_data = last.history(period="5d")
            
            if history_data.empty:
                st.warning(f"No data available for {stock}.")
            else:
                stock_prices = stock_data_df[stock_data_df['Stock Ticker'] == stock]['Close']
                volatility = calculate_volatility(stock_prices)
                volatility = f"{volatility:.2f}"    
                final = history_data['Close'][0]
                final = f"{final:.2f}"
            
                final = history_data['Close'][-1]
                final = f"{final:.2f}"
                
                CaP = get_call_price(stock)
                CaP = f"{CaP:.2f}"
                Put_price = get_put_price(stock) 
                Put_price = f"{Put_price:.2f}"

                circle_html = f"""<div class="circle">
                    <div class="content">{stock}</div>
                    <div>
                        Current Price: {final}<br>
                        Volatility: {volatility}<br>
                        Call Price: {CaP}<br>
                        Put Price: {Put_price}
                    </div>
                </div>
                """
                html_content += circle_html

        html_content += '</div>'

        st.markdown(css + html_content, unsafe_allow_html=True)

    else:
        st.warning("Please select at least one stock.")

elif page == "Portfolio Optimization":
    time.sleep(2)
    st.header('Portfolio Optimization Using The Markowitz Model', divider='gray')
    end_date = datetime.today()
    end_date_str = datetime.today().strftime('%Y-%m-%d')
    start_date = end_date - timedelta(days=7*365)  
    start_date_str = start_date.strftime('%Y-%m-%d')

    def is_valid_ticker(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if info.get('symbol') is not None:
                return True
        except Exception as e:
            return False
        return False

    tickers = st_tags(
        label="Enter one stock ticker at a time.",
        text="Press enter to add more",
        value=["AAPL", "MSFT"], 
    )


    selected_stocks = []
    for ticker in tickers:
        if is_valid_ticker(ticker):
            selected_stocks.append(ticker)
        else:
            st.warning(f"{ticker} is not a valid ticker symbol.")
    

    all_stock_data = pd.DataFrame()

    
    for ticker in selected_stocks:
        stock_df = yf.download(ticker, start=start_date_str, end=end_date_str, progress=False)
        if stock_df.empty:
            st.warning(f"No data available for {ticker}.")
        else:
            all_stock_data[ticker] = stock_df['Adj Close']

    mu = expected_returns.mean_historical_return(all_stock_data)

    filtered_stocks = {ticker: all_stock_data[ticker] for ticker in all_stock_data if mu[ticker] > 0}

    if len(filtered_stocks) == 0:
        if len(selected_stocks) == 0:
            st.warning("Please select at least one stock.")
        else:
            st.error("No stocks have positive expected returns. Optimal portfolio includes no stock.")
    else:

        S = risk_models.sample_cov(all_stock_data)

        ef = EfficientFrontier(mu, S)

        weights = ef.max_sharpe()

        clean_weights = ef.clean_weights()

        weights_df = pd.DataFrame(list(clean_weights.items()), columns=['Ticker', 'Weight'])
        weights_df = weights_df[weights_df['Weight'] > 0]  

        weights_df = weights_df.sort_values(by='Weight', ascending=False)
            
   
        sorted_tickers = weights_df['Ticker'].tolist()
        sorted_weights = weights_df['Weight'].tolist()
        
        plt.clf()  
        fig, ax = plt.subplots()

        wedges, texts, autotexts = ax.pie(
            sorted_weights,
            labels=sorted_tickers,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.get_cmap('tab10').colors,
            pctdistance=0.5
        )

        ax.legend(
            wedges, 
            sorted_tickers,
            title="Tickers",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1) 
        )
        for text in texts:
            text.set_fontsize(12)
            text.set_color('white')
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')

        ax.axis('equal')  
        ax.set_facecolor('none')  
        fig.patch.set_alpha(0)    


        st.subheader("Suggested percent of each stock in your portfolio:")


        st.pyplot(fig)
    

    

