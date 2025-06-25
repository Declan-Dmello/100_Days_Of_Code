import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a simple Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Create and train AdaBoost
adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),
                              n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)

# Make predictions
dt_pred = dt.predict(X_test)
adaboost_pred = adaboost.predict(X_test)

# Print accuracy scores
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("AdaBoost Accuracy:", accuracy_score(y_test, adaboost_pred))

# Print classification reports
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred))

print("\nAdaBoost Classification Report:")
print(classification_report(y_test, adaboost_pred))

# visualizing how each approach views the features
dt_importances = dt.feature_importances_
adaboost_importances = adaboost.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(range(20), dt_importances, alpha=0.5, label='Decision Tree')
plt.bar(range(20), adaboost_importances, alpha=0.5, label='AdaBoost')
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.title('Feature Importance: Decision Tree vs AdaBoost')
plt.legend()
plt.tight_layout()
plt.show()