import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from random_forest import RandomForest

bc = load_breast_cancer()
X,y = bc.data , bc.target
X_train, X_test , y_train, y_test =  train_test_split(X,y,test_size=0.2,random_state=1234)



clf = RandomForest(n_tress=100)
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred)/ len(y_test)

acc = accuracy(y_test,prediction)
print(acc)
