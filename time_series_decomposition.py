import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose



url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

print(df.head().to_string())

result = seasonal_decompose(df['Passengers'], period=12)
plt.figure(figsize=(15, 9))

plt.subplot(4, 1, 1)
plt.plot(df.index, df['Passengers'])
plt.title('Original Series Monthly Airline Passengers')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')

plt.subplot(4, 1, 2)
plt.plot(df.index, result.trend)
plt.title('Trend')
plt.xlabel('Year')
plt.ylabel('Trend')

plt.subplot(4, 1, 3)
plt.plot(df.index, result.seasonal)
plt.title('Seasonal')
plt.xlabel('Year')
plt.ylabel('Seasonality')

plt.subplot(4, 1, 4)
plt.plot(df.index, result.resid)
plt.title('Residual (Noise)')
plt.xlabel('Year')
plt.ylabel('Residuals')

plt.tight_layout()
plt.show()

print("\nTrend Analysis:")
print(result.trend.describe())
print("\nSeasonal Component Analysis:")
print(result.seasonal.describe())
print("\nResidual Component Analysis:")
print(result.resid.describe())