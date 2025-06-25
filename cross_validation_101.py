from sklearn.model_selection import KFold, StratifiedKFold,cross_val_score, TimeSeriesSplit
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)


rf_model = RandomForestClassifier(random_state=42)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(rf_model, X, y, cv=kfold)
print(f"K-Fold CV Scores: {kfold_scores.mean():.3f} ")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = cross_val_score(rf_model, X, y, cv=skf)
print(f"Stratified K-Fold CV Scores: {skf_scores.mean():.3f}")

tscv = TimeSeriesSplit(n_splits=5)
ts_scores = cross_val_score(rf_model, X, y, cv=tscv)
print(f"Time Series CV Scores: {ts_scores.mean():.3f}")






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


rf_model.fit(X_train, y_train)

y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()




precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='darkorange', lw=2,
         label=f'Precision-Recall curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()


def evaluate_binary_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba),
        'Average Precision': average_precision_score(y_test, y_pred_proba)
    }

    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return results


results = evaluate_binary_classifier(rf_model, X_test, y_test)
