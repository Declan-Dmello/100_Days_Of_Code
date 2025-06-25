import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')



def checking_the_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF :', result[0])
    print('the p value:', result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

print("Before differencing:")
checking_the_stationarity(df['Passengers'])

df['Passengers_diff'] = df['Passengers'].diff()
df = df.dropna()

print("\nAfter differencing:")
checking_the_stationarity(df['Passengers_diff'])




model = ARIMA(df['Passengers'], order=(1,1,1))
results = model.fit()

predictions = results.predict(start=len(df)-12, end=len(df)+11)

plt.figure(figsize=(12,6))
plt.plot(df.index, df['Passengers'], label='Original')
plt.plot(predictions.index, predictions, label='Predicted', color='red')
plt.title('ARIMA Model Predictions')
plt.legend()
plt.show()

print("\nModel Summary:")
print(results.summary())

rmse = np.sqrt(mean_squared_error(df['Passengers'][-12:], predictions[:12]))
print(f"\nRoot Mean Square Error: {rmse}")

print("\nForecast for next 12 months:")
print(predictions[-12:])