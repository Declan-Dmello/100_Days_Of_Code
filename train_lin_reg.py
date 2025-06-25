import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from Linear_reg import LinearRegression

X , y = datasets.make_regression(n_samples=1000,n_features=1,noise=20,random_state=4)
X_train , X_test , y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0], y,color="b", marker="o", s=30)
plt.show()

reg_model = LinearRegression()
reg_model.fit(X_train,y_train)
prediction  = reg_model.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test- prediction)**2)

mse = mse(y_test,prediction)
print(mse)


y_pred_line = reg_model.predict(X)
cmap =  plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train,y_train,c=cmap(0.9),s=10)
m2 = plt.scatter(X_test,y_test,c=cmap(0.5),s=10)
plt.plot(X,y_pred_line,color="black")
plt.show()