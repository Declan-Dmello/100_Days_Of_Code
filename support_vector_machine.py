import numpy as np
from sklearn.metrics import accuracy_score


class SVM:
    def __init__(self, learning_rate = 0.001, lambda_param = 0.01, n_iter = 1000):
        self.lr = learning_rate
        self.lambda_param= lambda_param
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X ,y):
        n_samples , n_features = X.shape
        y_ = np.where(y<=0, -1 ,1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iter):
            for idx , x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i , self.w) - self.b) >= 1
                if condition:
                    self.w  -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2* self.lambda_param * self.w - np.dot(x_i , y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X,self.w) - self.b
        return np.sign(approx)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    X, y = make_blobs(n_samples = 50 , n_features = 2, centers = 2,cluster_std =1.05,random_state=40)
    y = np.where(y==0, -1, 1)

    X_train , X_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state = 123)

    clf = SVM()
    clf.fit(X_train , y_train)
    prediction = clf.predict(X_test)
    acc = accuracy_score(y_test, prediction)
    print(acc)
