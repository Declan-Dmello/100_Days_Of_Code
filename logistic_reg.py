
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


class LogisticRegression():
    def __init__(self, lr= 0.005, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            linear_predictions =  np.dot(X , self.weights + self.bias)
            predictions =  sigmoid(linear_predictions)

            dw = (1/n_samples) * np.dot(X.T,(predictions -y))
            db = (1/n_samples) * sum(predictions - y )

            self.weights = self.weights - self.lr *  dw
            self.bias = self.bias - self.lr *  db

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights + self.bias)
        y_pred = sigmoid(linear_predictions)
        class_pred  = [0 if  y <= 0.5 else 1 for y in y_pred]
        return class_pred