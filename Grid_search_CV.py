import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import mlflow
import mlflow.sklearn



# Set the tracking URI to point to the server
mlflow.set_tracking_uri("http://localhost:5000")

# Set the experiment name
mlflow.set_experiment("Iris SVM Experiment")



iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC()
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Start an MLflow run
with mlflow.start_run():
    grid_search.fit(X_train, y_train)

    # Log the best parameters and best score
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cross_val_accuracy", grid_search.best_score_)

    # Train the model with the best hyperparameters
    best_model = grid_search.best_estimator_

    # Evaluate the model on the test set
    test_accuracy = best_model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Log the model
    mlflow.sklearn.log_model(best_model, "model")

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy: ", grid_search.best_score_)
    print("Test set accuracy: ", test_accuracy)

    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test, y_test)
    print("Test set accuracy: ", test_accuracy)
