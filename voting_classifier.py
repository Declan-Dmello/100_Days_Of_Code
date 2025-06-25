from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost_demo import XGBClassifier
from sklearn.datasets import make_classification



X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(random_state = 42)
model.fit(X_train, y_train)

model1 = RandomForestClassifier(random_state = 42)
model1.fit(X_train, y_train)

model2 = SVC(random_state = 42,probability=True)
model2.fit(X_train, y_train)

# Assuming you've calculated individual accuracies
xgb_accuracy = accuracy_score(y_test, model.predict(X_test))
rf_accuracy  = accuracy_score(y_test, model1.predict(X_test))
svc_accuracy = accuracy_score(y_test, model2.predict(X_test))

ensemble_weighted_soft = VotingClassifier(estimators=[
    ("xgb", model),
    ("rf", model1),
    ("svc", model2)],
    voting="soft",
    weights=[xgb_accuracy, rf_accuracy, svc_accuracy]
)

ensemble_weighted_hard = VotingClassifier(estimators=[
    ("xgb", model),
    ("rf", model1),
    ("svc", model2)],
    voting="hard",
    weights=[xgb_accuracy, rf_accuracy, svc_accuracy]
)

ensemble_weighted_soft.fit(X_train, y_train)
prediction_weighted = ensemble_weighted_soft.predict(X_test)
print("Ensemble soft (Weighted Voting) Accuracy:", accuracy_score(y_test, prediction_weighted))
ensemble_weighted_hard.fit(X_train, y_train)
prediction_weighted_hard = ensemble_weighted_hard.predict(X_test)
print("Ensemble hard (Weighted Voting) Accuracy:", accuracy_score(y_test, prediction_weighted_hard))


