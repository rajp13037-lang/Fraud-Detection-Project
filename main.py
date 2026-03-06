from src.preprocessing import load_data, preprocess_data, apply_smote
from src.train import train_logistic, train_random_forest, train_xgboost
from src.evaluate import evaluate_model
import joblib


def main():
    print("loading data..")
    df = load_data("data/creditcard.csv")

    print("preprocessing data..")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("applying smote..")
    X_resampled, y_resampled = apply_smote(X_train, y_train)

    print("training logistic regression..")
    model = train_logistic(X_resampled, y_resampled)

    print("evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("saving model...")
    joblib.dump(model, "models/fraud_model.pkl")

    print("done")


if __name__ == "__main__":
    main()