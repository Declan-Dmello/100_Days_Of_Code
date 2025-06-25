import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


bagging = BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1)
bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)
print(f"Bagging Accuracy: {accuracy_score(y_test, bagging_pred):.4f}")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.4f}")

#Adding a bit of noise to the synthetic dataset

np.random.seed(42)
X_noisy = X + np.random.normal(0, 0.1, X.shape)
X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(X_noisy, y, test_size=0.2, random_state=42)

bagging.fit(X_train_noisy, y_train_noisy)
dt.fit(X_train_noisy, y_train_noisy)

print("Testing performance on noisy data:")
print(f"Bagging Accuracy: {accuracy_score(y_test_noisy, bagging.predict(X_test_noisy)):.4f}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test_noisy, dt.predict(X_test_noisy)):.4f}")
