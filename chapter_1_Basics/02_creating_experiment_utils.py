import logging
import mlflow_utils.utils as utils

logger = logging.getLogger()

if __name__ == "__main__":
    # Creating an MLflow experiment from customized module
    experiment_id = utils.create_experiment(
        experiment_name="mlflow_demo1",
        artifact_location="mlflow_demo_artifacts1",
        tags={"env": "rnd", "version": "0.0.2"},
    )
    logger.info(f"Display the experiment id {experiment_id}")
    print("Experiment_id: {}".format(experiment_id))
