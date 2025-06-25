from xgboost import XGBRegressor
from sklearn.datasets import make_classification,make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from gradient_boosting import SimpleGradientBoosting

X, y = make_regression(n_samples=5000,n_features=20, noise=5,random_state=42)
X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2 ,random_state=42)

model = XGBRegressor()
model.fit(X_train,y_train)
prediction=model.predict(X_test)

model2 = SimpleGradientBoosting()
model2.fit(X_train,y_train)
prediction1 = model2.predict(X_test)

print(f"Mean Absolute Error : {mean_absolute_error(y_test,prediction)}")
print(f"Mean Absolute Error : {mean_absolute_error(y_test,prediction1)}")
