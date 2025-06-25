import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import shap
from xgboost_demo import XGBRegressor
from sklearn.metrics import mean_absolute_error


# Create a sample dataset

train_data = pd.DataFrame(pd.read_csv("../data/concrete_data.csv"))
y = train_data["Strength"]
X = train_data.drop(["Strength"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=12)

print(y)
print(X)


model = XGBRegressor()
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values,X_test)

print(shap_values)
