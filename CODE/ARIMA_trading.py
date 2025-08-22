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


data = data[['Close']]
data['daily_ret'] = data['Close'].diff()
data['daily_ret_norm'] = np.log(data['Close']).diff()
data = data.ffill()

data.dropna(inplace=True)

def arima_forecast(window_size,current_data):
    train_window = current_data.iloc[:window_size][['daily_ret_norm']].copy()
    predictions = pd.DataFrame(columns=['Actual', 'Forecast'])
    for i in range(window_size, len(current_data)):
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

        actual = current_data.iloc[i]['daily_ret_norm']

        predictions.loc[current_data.index[i]] = [actual, forecast[0]]

        train_window.loc[current_data.index[i]] = actual
        train_window = train_window.iloc[1:]
        #print(f"Processed {i+1}/{len(data)}: Actual={actual}, Forecast={forecast[0]}")
    predictions.index = pd.to_datetime(predictions.index) 

    # plt.figure(figsize=(12,6))
    # plt.plot(predictions['Forecast'], color='red', label='Predicted')
    # plt.plot(predictions['Actual'],  color='green', label='Actual')

    # plt.title('ARIMA Forecast vs Actual')
    # plt.xlabel('Date')
    # plt.ylabel('daily_ret_norm')
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig('PLOTS/ARIMA_trading/Forecast.png', dpi=300)
    #plt.close()

    #____________________________________________
    # Calculate returns and strategy performance
    #____________________________________________
    initial_capital=100.0
    # Ensure numeric
    predictions[['Actual','Forecast']] = predictions[['Actual','Forecast']].astype(float)

    # Signal και στρατηγική (shift για αποφυγή lookahead)
    predictions['Signal'] = (predictions['Forecast'] > 0).astype(int)
    predictions['Strategy_LogRet'] = predictions['Signal'].shift(1) * predictions['Actual']
    predictions['BuyHold_LogRet'] = predictions['Actual']

    # From log-returns to wealth
    predictions['CumLog_Strategy'] = predictions['Strategy_LogRet'].cumsum().fillna(0)
    predictions['Wealth_Strategy'] = initial_capital * np.exp(predictions['CumLog_Strategy'])

    predictions['CumLog_BuyHold'] = predictions['BuyHold_LogRet'].cumsum().fillna(0)
    predictions['Wealth_BuyHold'] = initial_capital * np.exp(predictions['CumLog_BuyHold'])

    # Plot wealth curves
    plt.figure(figsize=(12,6))
    plt.plot(predictions.index, predictions['Wealth_Strategy'], label='ARIMA Strategy Wealth')
    plt.plot(predictions.index, predictions['Wealth_BuyHold'], label='Buy & Hold Wealth')
    plt.legend()
    plt.grid(True)
    plt.title('ARIMA Strategy vs Buy & Hold (Wealth)')
    plt.tight_layout()
    plt.savefig(f'PLOTS/ARIMA_trading/ARIMA_Strategy_vs_BuyAndHold.png', dpi=300)
    plt.close()

    # Μικρό summary
    final_strategy = predictions['Wealth_Strategy'].iloc[-1]
    final_bh = predictions['Wealth_BuyHold'].iloc[-1]
    print(f"Final wealth ARIMA strategy={final_strategy:.2f} vs buy&hold={final_bh:.2f}, better? {final_strategy>final_bh}")

#___________________________
start_year = 1929
end_year = 1929
window_size = 100

data = data[((data.index.year >= start_year) & (data.index.year <= end_year))]

arima_forecast(window_size,data)



