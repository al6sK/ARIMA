import yfinance as yf

sp500 = yf.Ticker("^GSPC")
df = sp500.history(period="max" , interval="1d")
df.to_csv("DATA/S&P500_2025.csv")
