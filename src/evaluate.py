from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("\nROC AUC Score:", roc)