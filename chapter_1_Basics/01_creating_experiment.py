import mlflow
import logging

logger = logging.getLogger()

if __name__ == "__main__":
    # Creating an ML flow experiment
    experiment_id = mlflow.create_experiment(
        name="mlflow_demo",
        artifact_location="mlflow_demo_artifacts",
        tags={"env": "rnd", "version": "0.0.1"},
    )
    logger.info(f"Display the experiment id {experiment_id}")
