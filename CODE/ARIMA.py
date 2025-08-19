import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

#___________________
# Load the dataset
#___________________
df = pd.read_csv("DATA/S&P500_2025.csv")

print(df.head(20))


df.set_index('Date', inplace=True)


print(df.head())
print(df.shape)

#___________________
#Prepere data for ARIMA
#___________________
df['Daily_re'] = 