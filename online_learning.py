import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


class OnlineLearningModel:
    def __init__(self, learning_rate=0.01, loss='log_loss'):
        self.model = SGDClassifier(
            loss=loss,
            learning_rate='constant',
            eta0=learning_rate,
            max_iter=1000
        )
        self.scaler = StandardScaler()
        self.is_initialized = False

    def partial_fit(self, X, y, classes=None):
        if not self.is_initialized:
            X = self.scaler.fit_transform(X)
            self.model.partial_fit(X, y, classes=np.unique(y))
            self.is_initialized = True
        else:
            X = self.scaler.transform(X)
            self.model.partial_fit(X, y)

        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def score(self, X, y):
        X = self.scaler.transform(X)
        return self.model.score(X, y)


def demonstrate_online_learning():
    np.random.seed(42)

    X_initial = np.random.randn(100, 5)
    y_initial = np.random.randint(0, 2, 100)

    online_model = OnlineLearningModel()

    online_model.partial_fit(X_initial, y_initial, classes=[0, 1])

    for batch in range(7):
        X_new = np.random.randn(20, 5)
        y_new = np.random.randint(0, 2, 20)

        online_model.partial_fit(X_new, y_new)
        print(f"Batch {batch + 1} done --- Current model accuracy: {online_model.score(X_new, y_new):.2f}")


demonstrate_online_learning()