import mlflow
from typing import Dict
import logging

logger = logging.getLogger()


def create_experiment(
    experiment_name: str,
    artifact_location: str,
    tags: Dict[str, str]
) -> str:
    """_summary_

    Args:
        experiment_name (str): Name of the experiment
        artifact_location (str): Location of the created experiment
        tags (dict[str, Any]): Tags of the experiment

    Returns:
        experiment_id: str
        Id of the created experiment
    """
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags=tags
        )
    except Exception:
        logger.info(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name
        (experiment_name).experiment_id
    mlflow.set_experiment(experiment_name=experiment_name)
    return experiment_id
