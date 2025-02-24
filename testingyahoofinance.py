import yfinance as yf

# Define the stock ticker symbol
ticker_symbol = "AAPL"

# Create a Ticker object for the given symbol
stock = yf.Ticker(ticker_symbol)

# Fetch historical market data for the last month
historical_data = stock.history(period="1mo")
print("Historical Data (Last Month):")
print(historical_data)

# Fetch current market price from the stock's info dictionary
current_price = stock.info.get("regularMarketPrice", "N/A")
print(f"\nCurrent Market Price for {ticker_symbol}: {current_price}")