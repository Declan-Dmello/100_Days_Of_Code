import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

data = load_iris()
data_view = pd.DataFrame(data.data, columns=data.feature_names)

data_view['target'] = data.target

print(data_view.to_string())
X, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = LogisticRegression(max_iter=100)
model.fit(x_train, y_train)

pickle.dump(model, open('model1.pkl', 'wb'))