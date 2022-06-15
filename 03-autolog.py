import os

import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from preprocess import load_pickle, dump_pickle


def train_models_normal(data_path):
    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

        mlflow.set_tag("developer", "dulanj")

        mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
        mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")

        alpha = 0.2
        mlflow.log_param("alpha", alpha)
        lr = Lasso(alpha)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_valid)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        dv = load_pickle(os.path.join(data_path, "dv.pkl"))

        # create dest_path folder unless it already exists
        os.makedirs("models", exist_ok=True)

        dump_pickle((dv, lr), os.path.join("models", "lasso.pkl"))
        mlflow.log_artifact(local_path="models/lasso.pkl", artifact_path="models_pickle")


def train_model_autolog(data_path):
    with mlflow.start_run():
        mlflow.sklearn.autolog()

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        print(f"RMSE: {rmse}")


if __name__ == '__main__':
    # mlflow ui --backend-store-uri sqlite:///mlflow.db

    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # mlflow.set_experiment("Mflow-test-run")
    # train_models_normal(data_path="./processed")

    mlflow.set_experiment("Mflow-autolog")
    train_model_autolog(data_path="./processed")
