import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://localhost:5000"
# MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# EXPERIMENT_NAME = "Mflow-XGBoost"
EXPERIMENT_NAME = "Mflow-RandomForest"


def get_list_of_models(client, model_name):
    latest_versions = client.get_latest_versions(name=model_name)
    print("*" * 80)
    for version in latest_versions:
        print(f"version: {version.version}, stage: {version.current_stage}")


def register_a_model(model_name, run_id):
    # # register the best model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    print(f"registering model: {model_uri}")
    mlflow.register_model(model_uri=model_uri, name=model_name)


def model_staging(model_name, model_version, new_stage):
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=True
    )


def main():
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    metric_name = "training_rmse"

    # select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string='attributes.status != "FAILED"',
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=2,
        order_by=[f'metrics."{metric_name}" ASC']
    )[0]
    print(f"run id: {best_run.info.run_id}, rmse: {best_run.data.metrics[f'{metric_name}']:.4f}")

    model_name = "NY-Taxi-Model"

    # Register the best model
    print("Registering model:")
    run_id = best_run.info.run_id
    register_a_model(model_name, run_id)

    get_list_of_models(client, model_name)

    # Model staging
    print("Model staging:")
    model_version = 2
    new_stage = "Staging"
    model_staging(model_name, model_version, new_stage)

    get_list_of_models(client, model_name)


if __name__ == '__main__':
    main()
