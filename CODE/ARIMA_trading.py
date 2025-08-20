import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
import warnings

warnings.filterwarnings("ignore")

#______________________________________
# Load the dataset
#______________________________________

data = pd.read_csv("DATA/S&P500_2025.csv")
data.set_index('Date', inplace=True)
data.index = pd.to_datetime(data.index, utc=True)

#______________________________________
# Ready the data for ARIMA
#______________________________________
start_year = 2000
end_year = 2000
data = data[(data.index.year >= start_year) & (data.index.year <= end_year)]

data = data[['Close']]
data['daily_ret'] = data['Close'].pct_change()
data['daily_ret_norm'] = np.log(data['Close']).diff()
data = data.ffill()

data.dropna(inplace=True)

window_size = 200
train_window = data.iloc[:window_size][['daily_ret_norm']].copy()

predictions = pd.DataFrame(columns=['Actual', 'Forecast'])


for i in range(window_size, len(data)):
    acf_values = acf(train_window.values, nlags=4)
    acf_values = acf_values[1:] # skip lag=0
    max_lag_acf = np.argmax(acf_values) + 1  

    pacf_values = pacf(train_window.values, nlags=4)
    pacf_values = pacf_values[1:]  # skip lag=0
    max_lag_pacf = np.argmax(pacf_values) + 1  
    p = max_lag_acf
    q = max_lag_pacf

    model = ARIMA(train_window.values, order = (p, 0, q)).fit()
    forecast = model.forecast(steps=1)

    actual = data.iloc[i]['daily_ret_norm']

    predictions.loc[data.index[i]] = [actual, forecast[0]]

    train_window.loc[data.index[i]] = actual
    if len(train_window) > window_size:
        train_window = train_window.iloc[1:]
    print(f"Processed {i+1}/{len(data)}: Actual={actual}, Forecast={forecast[0]}")


predictions.index = pd.to_datetime(predictions.index)

idx = pd.date_range(predictions.index.min(), predictions.index.max(), freq='B') 
df_cont = predictions.reindex(idx)   

df_cont = df_cont.ffill()   


plt.figure(figsize=(12,6))
plt.plot(predictions['Forecast'], color='red', label='Predicted')
plt.plot(predictions['Actual'],  color='green', label='Actual')

plt.title('ARIMA Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('daily_ret_norm')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('PLOTS/ARIMA_trading/Forecast.png', dpi=300)


#____________________________________________
# Calculate returns and strategy performance
#____________________________________________

initial_capital = 1  # Ή ό,τι κεφάλαιο θες

# Δημιουργούμε τα σήματα
predictions['Order'] = (predictions['Forecast'] > 0).astype(int)

# Κέρδος ανά ημέρα με βάση το signal
predictions['Profit'] = predictions['Order'] * data['daily_ret']

# Κεφάλαιο με βάση ARIMA strategy
predictions['ARIMA_Capital'] = initial_capital * (1 + predictions['Profit']).cumprod()

# Κεφάλαιο Buy & Hold (επένδυση στην πρώτη μέρα και αφήνουμε να τρέξει)
predictions['BuyAndHold'] = initial_capital * (1 + data['daily_ret']).cumprod()

print("Final capital with ARIMA strategy:", predictions['ARIMA_Capital'].iloc[-1])
print("Final capital with Buy & Hold:", predictions['BuyAndHold'].iloc[-1])

predictions['Difference'] = predictions['ARIMA_Capital'] - predictions['BuyAndHold']


plt.figure(figsize=(12,6))

# Plot ARIMA vs Buy & Hold
plt.plot(predictions.index, predictions['ARIMA_Capital'], label='ARIMA Strategy', color='blue')
plt.plot(predictions.index, predictions['BuyAndHold'], label='Buy & Hold', color='green')

# Plot Difference
plt.plot(predictions.index, predictions['Difference'], label='Difference (ARIMA - Buy&Hold)', color='red', linestyle='--')

plt.title("Capital Comparison: ARIMA vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Capital (€)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('PLOTS/ARIMA_trading/Strategy_vs_BuyAndHold.png', dpi=300)
