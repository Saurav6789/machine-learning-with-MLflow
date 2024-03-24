import mlflow
from mlflow_utils import utils

experiment_id = utils.create_experiment(experiment_name="Nested_runs",
                                        artifact_location="nested_artifiacts",
                                        tags={"nested_runs": "example"})


if __name__ == "__main__":
    print("Experiment_id", experiment_id)
    with mlflow.start_run(run_name="Project") as project_run:
        print("RUN ID parent:", project_run.info.run_id)
        mlflow.log_param("project_param", "project_value")

        # Nested run for the data preprocessing
        with mlflow.start_run(run_name="Data Preprocessing",
                              nested=True) as data_run:
            print("RUN ID Data Preprocessing:", data_run.info.run_id)
            mlflow.log_param("preprocess_param", "preprocess_value")

        # Nested run for the feature engineering
        with mlflow.start_run(run_name="Feature Engineering",
                              nested=True) as feature_run:
            print("RUN ID Feature Engineering:", feature_run.info.run_id)
            mlflow.log_param("feature_param", "feature_value")

        # Nested run for the model building
        with mlflow.start_run(run_name="Model Building",
                              nested=True) as model_run:
            print("RUN ID Feature Engineering:", model_run.info.run_id)
            mlflow.log_param("model_param", "model_value")

        # Nested run for the model building
        with mlflow.start_run(run_name="Model Evaluation",
                              nested=True) as model_eval:
            print("RUN ID Feature Engineering:", model_eval.info.run_id)
            mlflow.log_param("modelEval_param", "modelEval_value")
