import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA

#___________________
# Load the dataset
#___________________
df = pd.read_csv("DATA/S&P500_2025.csv")

df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index, utc=True)

#___________________
# Prepere data for ARIMA
#___________________

df_week = df.resample('W').mean()
df_week = df_week[['Close']]

df_week['weekly_ret'] = np.log(df_week['Close']).diff()

df_week.dropna(inplace=True)

df_week.weekly_ret.plot(kind= "line" , figsize=(12, 6))
plt.savefig('PLOTS/ARIMA/data_normalization.png', dpi=300)

udiff = df_week.drop(['Close'], axis=1)
#___________________
# Test for stationarity of the udiff series
#___________________

rolmean = udiff.rolling(20).mean()
rolstd = udiff.rolling(20).std()

plt.figure(figsize=(12,6))
orig = plt.plot(udiff, color="blue", label="Original")
mean = plt.plot(rolmean, color="red", label="Rolling Mean")
std = plt.plot(rolstd, color="black", label="Rolling Std Deviation")
plt.title("Rolling Mean & Standard Deviation")
plt.legend(loc="best")
plt.savefig('PLOTS/ARIMA/Rolling_Mean_&_Standard_Deviation.png', dpi=300)

# Perform Dickey-Fuller test
dftest = sm.tsa.adfuller(udiff.weekly_ret , autolag="AIC")
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value ({0})'.format(key)] = value
print(dfoutput)

# the autocorrelation chart provides just the correlation at increasing lags
fig, ax = plt.subplots(figsize=(12,5))
plot_acf(udiff.values, lags=10, ax=ax)
plt.savefig('PLOTS/ARIMA/autocorrelation_chart.png', dpi=300)


fig, ax = plt.subplots(figsize=(12,5))
plot_pacf(udiff.values, lags=10, ax=ax)
plt.savefig('PLOTS/ARIMA/partial_autocorrelation_chart.png', dpi=300)

#___________________
# Build ARIMA Model
#___________________

ar1 = ARIMA(udiff.values, order = (1, 0, 1)).fit()
print(ar1.summary())

plt.figure(figsize=(12, 8))
plt.plot(udiff.values, color='blue')
preds = ar1.fittedvalues
plt.plot(preds, color='red')
plt.savefig('PLOTS/ARIMA/Actual_vs_Fitted_Values.png', dpi=300)


#___________________
# Forecast 
#___________________

steps = 2

forecast = ar1.forecast(steps=steps)

plt.figure(figsize=(12, 8))
plt.plot(udiff.values, color='blue')

preds = ar1.fittedvalues
plt.plot(preds, color='red')

plt.plot(pd.DataFrame(np.array([preds[-1],forecast[0]]).T,index=range(len(udiff.values)+1, len(udiff.values)+3)), color='green')
plt.plot(pd.DataFrame(forecast,index=range(len(udiff.values)+1, len(udiff.values)+1+steps)), color='green')
plt.title('Display the predictions with the ARIMA model')
plt.savefig('PLOTS/ARIMA/Forecast.png', dpi=300)

