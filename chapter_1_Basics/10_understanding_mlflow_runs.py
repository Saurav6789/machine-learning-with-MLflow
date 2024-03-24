import mlflow
from mlflow_utils import utils
import matplotlib.pyplot as plt


if __name__ == "__main__":
    experiment = utils.retrieve_experiment(experiment_id="291155198390524433")
    print("Name: {}".format(experiment.name))
    with mlflow.start_run(run_name="mlflow_test_runs",
                          experiment_id="291155198390524433") as mlflow_run:
        parameters = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_epochs": 10
        }
        mlflow.log_params(parameters)
        # log a metric
        mlflow.log_metric("accuracy", 0.78)

        # Log an artifact (e.g., a file)
        with open("example.txt", "w") as file:
            file.write("This is an example artifact.")
        mlflow.log_artifact("example.txt")

        # Log an image
        plt.plot([1, 2, 3, 4])
        plt.ylabel('some numbers')
        plt.savefig("plot.png")
        mlflow.log_artifact("plot.png", artifact_path="images")

        # Display the run info
        print("run_id: {}".format(mlflow_run.info.run_id))
        print("experiment_id: {}".format(mlflow_run.info.experiment_id))
        print("status: {}".format(mlflow_run.info.status))
        print("start_time: {}".format(mlflow_run.info.start_time))
        print("end_time: {}".format(mlflow_run.info.end_time))
        print("lifecycle_stage: {}".format(mlflow_run.info.lifecycle_stage))
