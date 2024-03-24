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


def retrieve_experiment(
    experiment_id: str = None,
    experiment_name: str = None
) -> mlflow.entities.Experiment:
    """_summary_

    Args:
        experiment_id (str, optional):
        The Id of the experiment.
        experiment_name (str, optional):
        The name of the experiment

    Returns:
        experiment: mlflow.entities.Experiment
         The mlflow experiment with the given id or name.
    """
    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError(
            "Both experiment_id and experiment_name is not provided")
    return experiment
