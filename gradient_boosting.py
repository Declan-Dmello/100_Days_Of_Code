import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

class SimpleGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.01):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        current_predictions = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residuals = y - current_predictions
            model = DecisionTreeRegressor(max_depth=3)
            model.fit(X, residuals)
            self.models.append(model)
            current_predictions += self.learning_rate * model.predict(X)

    def predict(self, X):
        return sum(self.learning_rate * model.predict(X) for model in self.models)


gb = SimpleGradientBoosting()
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gb.fit(X_train, y_train)
predictions = gb.predict(X_test)
#print(mean_absolute_error(y_test,predictions))