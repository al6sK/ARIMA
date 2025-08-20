import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

#______________________________________
# Load the dataset
#______________________________________

data = pd.read_csv("DATA/S&P500_2025.csv")
data.set_index('Date', inplace=True)
data.index = pd.to_datetime(data.index, utc=True)

#______________________________________
# Split data to Train and Test
#______________________________________

data.drop(['High','Low','Volume','Dividends','Stock Splits'], axis=1, inplace=True)
data['daily_ret'] = data['Close'].diff()
data['daily_ret_norm'] = np.log(data['Close']).diff()
data.dropna(inplace=True)

start_year = 1928
end_year = 1930
train = data[(data.index.year >= start_year) & (data.index.year <= end_year)]
train = train[['daily_ret_norm']]

#___________________________________________
# Test for stationarity of the train series
#___________________________________________

rolmean = train.rolling(10).mean()
rolstd = train.rolling(10).std()

plt.figure(figsize=(12,6))
orig = plt.plot(train, color="blue", label="Original")
mean = plt.plot(rolmean, color="red", label="Rolling Mean")
std = plt.plot(rolstd, color="black", label="Rolling Std Deviation")
plt.title("Rolling Mean & Standard Deviation")
plt.legend(loc="best")
plt.savefig('PLOTS/ARIMA_on_daily_ret/Rolling_Mean_&_Standard_Deviation.png', dpi=300)

# Perform Dickey-Fuller test
dftest = sm.tsa.adfuller(train.daily_ret_norm , autolag="AIC")
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value ({0})'.format(key)] = value
print(dfoutput)

# Finding correlation 
acf_values = acf(train.values, nlags=4)
acf_values = acf_values[1:] # skip lag=0
max_lag_acf = np.argmax(acf_values) + 1  
print("max_lag_acf όπου εμφανίζεται:", max_lag_acf)

# Finding partial autocorrelation
pacf_values = pacf(train.values, nlags=4)
pacf_values = pacf_values[1:]  # skip lag=0
max_lag_pacf = np.argmax(pacf_values) + 1  

print("max_lag_pacf όπου εμφανίζεται:", max_lag_pacf)

p = max_lag_acf
q = max_lag_pacf

#___________________
# Build ARIMA Model
#___________________

ar1 = ARIMA(train.values, order = (p, 0, q)).fit()
print(ar1.summary())

plt.figure(figsize=(12, 8))
plt.plot(train.values, color='blue')
preds = ar1.fittedvalues
plt.plot(preds, color='red')
plt.savefig('PLOTS/ARIMA_on_daily_ret/Actual_vs_Fitted_Values.png', dpi=300)

#___________________
# Forecast 
#___________________

steps = 10

forecast = ar1.forecast(steps=steps)

df_forecast = pd.DataFrame({
    'forecast': forecast
})

next = data[data.index.year == end_year + 1]['daily_ret_norm'].iloc[:steps]

plt.figure(figsize=(12, 6))

plt.plot(train.index, train.values, label='Actual data', color='blue')
plt.plot(train.index, preds, label='Fitted values', color='red')

forecast_index = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
plt.plot([train.index[-1], forecast_index[0]], [preds[-1], forecast[0]], color='green', linestyle='--')
plt.plot(forecast_index, forecast, label='Forecast', color='green')

plt.plot(forecast_index[:len(next)], next.values, label='Real values', color='black')

plt.title('ARIMA Model: Actual vs Fitted vs Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('PLOTS/ARIMA_on_daily_ret/Forecast.png', dpi=300)


next = pd.DataFrame({
    'daily_ret_norm': next.values
}, index=next.index)

df_forecast['Order'] =[1 if sig > 0 else -1 for sig in df_forecast['forecast'].diff()]
next['Order'] =[1 if sig > 0 else -1 for sig in next['daily_ret_norm'].diff()]

# print(df_forecast.head(10))
# print(next.head(10))