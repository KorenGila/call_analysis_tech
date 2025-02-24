import yfinance as yf

ticker = "AAPL"

try:
    stock = yf.Ticker(ticker)
    data = stock.history(period="1mo")
    print("Historical Data:")
    print(data)
    print("Current Price:", stock.info.get("regularMarketPrice", "N/A"))
except Exception as e:
    print("Error fetching data:", e)
