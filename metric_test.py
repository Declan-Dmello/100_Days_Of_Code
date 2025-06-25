import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from logistic_reg import LogisticRegression
from sklearn.linear_model import LinearRegression
from class_reg_metrics import accuracy,confusion_matrix1, mae,rmse
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, root_mean_squared_error


bc = datasets.load_breast_cancer()
X, y = bc.data,bc.target
X_train, X_test , y_train, y_test = train_test_split(X, y , train_size=0.8, random_state= 123)

model = LogisticRegression()

model.fit(X_train,y_train)

prediction = model.predict(X_test)


score_dev = accuracy(prediction,y_test)
print(f"Created Accuracy score{score_dev}")
score_imp = accuracy_score(y_test,prediction)
print(f"Imported Accuracy Score{score_imp}")

print("\n")

cm = confusion_matrix1(y_test,prediction, 2)
print(f" Created Confusion Matrix : {cm}")


i_cm = confusion_matrix(y_test, prediction)
print(f"Imported Confusion Matrix: {i_cm}")

print("\n")



X, y = datasets.make_regression(n_samples=1000,n_features=1,noise=20,random_state=4)
X_train, X_test , y_train, y_test = train_test_split(X, y , train_size=0.8, random_state= 123)

model1 = LinearRegression()

model1.fit(X_train,y_train)

prediction1 = model1.predict(X_test)



reg_score = mae(y_test,prediction1)
reg_score1 = rmse(y_test,prediction1)

imp_reg_score = mean_absolute_error(y_test,prediction1)
imp_reg_score1 = root_mean_squared_error(y_test,prediction1)

print(f"Created Mean Absolute Error  :{reg_score}")
print(f"Imported Mean Absolute Error  :{imp_reg_score}")

print(f"Created Root Mean Squared Error :{reg_score1}")
print(f"Imported Root Mean Squared Error : {imp_reg_score1}")








