import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
y = housing.target[:100]

dates = pd.date_range(start='2022-01-01', periods=len(y), freq='D')
df = pd.DataFrame({
    'ds': dates,
    'y': y
})
model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False
)
model.fit(df)

future = model.make_future_dataframe(periods=15)
forecast = model.predict(future)

plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], 'b.', label='Actual Prices')
plt.plot(forecast['ds'], forecast['yhat'], 'r-', label='Predicted Prices')
plt.fill_between(forecast['ds'],
                 forecast['yhat_lower'],
                 forecast['yhat_upper'],
                 color='r',
                 alpha=0.1,
                 label='Uncertainty')

plt.title('California Housing Prices->> Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price (100k$)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nForecast of the next 5 Days:")
future_5 = forecast[['ds', 'yhat']].tail(5)
future_5['ds'] = future_5['ds'].dt.strftime('%Y-%m-%d')
future_5.columns = ['Date', 'Predicted Price ($100k)']
print(future_5.to_string(index=False))
