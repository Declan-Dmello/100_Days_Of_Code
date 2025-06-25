import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR

from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import GridSearchCV


# Create a sample dataset

train_data = pd.DataFrame(pd.read_csv("../data/concrete_data.csv"))
y = train_data["Strength"]
X = train_data.drop(["Strength"], axis =1)


X_train , X_test , y_train, y_test = train_test_split(X,y,train_size=0.8, random_state=12)

print(y)
print(X)


#using SMOTE

# Split the data into training and testing sets

# Define the search space
search_spaces_bs = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'gamma': Real(1e-6, 1e+1, prior='log-uniform')

    
}

search_spaces_gs = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'gamma': Real(1e-6, 1e+1, prior='log-uniform')

}

# Create and run the optimizer
Bayes_search = BayesSearchCV(
    SVR(),
    search_spaces_bs,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=1
)


model = SVR()
model.fit(X_train,y_train)
pred  =  model.predict(X_test)
Bayes_search.fit(X_train, y_train)
best_model = Bayes_search.best_estimator_
y_pred = best_model.predict(X_test)




acc_score = mean_absolute_error(y_test,pred)
test_accuracy = mean_absolute_error(y_test, y_pred)


print(f"MAE without any Hyperpara Tuning: {acc_score}")

print(f"MAE with BayesSearchCV: {test_accuracy}")