import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import time

X, y = make_regression(n_samples=5000,n_features=20, noise=5,random_state=42)
X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2 ,random_state=42)

lgbm = LGBMRegressor(n_estimators=100, random_state=42,force_col_wise=True)
xgb = XGBRegressor(n_estimators=100, random_state=42)

s_t = time.time()
lgbm.fit(X_train, y_train)
lgbm_train_time = time.time() - s_t

lgbm_pred = lgbm.predict(X_test)
lgbm_mse = mean_squared_error(y_test, lgbm_pred)
lgbm_r2 = r2_score(y_test, lgbm_pred)

s_t = time.time()
xgb.fit(X_train, y_train)
xgb_train_time = time.time() - s_t

xgb_pred = xgb.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score( y_test, xgb_pred)

print("LightGBM:")
print(f"Training Time : {lgbm_train_time:.4f} secs")
print(f"Mean Squared Error: {lgbm_mse}")
print(f"The R-Sqrd Score: {lgbm_r2}")

print("XGBoost:")
print(f"Training Time : {xgb_train_time:.4f} secs")
print(f"Mean Squared Error: {xgb_mse}")
print(f"The R-Sqrd Score: {xgb_r2}")