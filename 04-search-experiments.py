import os

import mlflow
import numpy as np
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from preprocess import load_pickle

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def random_forest_training(data_path):
    mlflow.set_experiment("Mflow-RandomForest")

    mlflow.sklearn.autolog()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    def objective(params):
        with mlflow.start_run():
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_valid)
            rmse = mean_squared_error(y_valid, y_pred, squared=False)

            mlflow.log_param("rmse", rmse)

            return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials(),
        rstate=rstate
    )


def xgb_training(data_path):
    mlflow.set_experiment("Mflow-XGBoost")

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    mlflow.xgboost.autolog()

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_valid, label=y_valid)

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            # mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=10,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_valid, y_pred, squared=False)
            # mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials()
    )


def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    random_forest_training(data_path="./processed")
    # xgb_training(data_path="./processed")


if __name__ == '__main__':
    # train()

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp_list = client.list_experiments()

    for exp in exp_list:
        print("Experiment:", exp.experiment_id, exp.name)

    metric_name = "training_rmse"
    runs = client.search_runs(
        experiment_ids=['4'],
        filter_string=f'metrics."{metric_name}" < 8.5 and attributes.status != "FAILED"',
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=3,
        order_by=[f'metrics."{metric_name}" ASC']
    )

    print("\nBest runs:")
    for run in runs:
        print(f"run id: {run.info.run_id}, {metric_name}: {run.data.metrics[f'{metric_name}']:.4f}")
