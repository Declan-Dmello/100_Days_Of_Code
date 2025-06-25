import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from k_nn import KNN

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()


X,y = iris.data , iris.target

X_train, X_test , y_train, y_test = train_test_split(X, y , train_size=0.8, random_state=1234)


plt.figure()
plt.scatter(X[:,2],X[:,3], c=X[:,0], edgecolor = 'k', s=20)
plt.show()
clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions  = clf.predict(X_test)

print(predictions)

acc = accuracy_score(y_test,predictions)
print(acc)