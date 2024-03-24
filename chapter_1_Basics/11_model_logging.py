import mlflow
from mlflow.models import infer_signature
from mlflow_utils import utils
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    experiment = utils.retrieve_experiment(experiment_id="291155198390524433")
    print("Name: {}".format(experiment.name))
    with mlflow.start_run(run_name="model_loogging",
                          experiment_id="291155198390524433") as mlflow_run:
        # Load the iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42)

        # Using the logistic Regression Model
        model_logit = LogisticRegression(max_iter=1000)
        model_logit.fit(X_train, y_train)
        y_pred = model_logit.predict(X_test)

        # infer signature
        model_signature = infer_signature(
            model_input=X_train, model_output=y_pred)

        # log model
        mlflow.sklearn.log_model(model_logit, "Logistic Regression")
