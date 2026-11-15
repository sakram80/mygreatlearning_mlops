import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load the four split files downloaded by the workflow
X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")
y_train = pd.read_csv("ytrain.csv").squeeze()
y_test = pd.read_csv("ytest.csv").squeeze()

# 2. Build preprocessing + model pipeline (compatible with pandas 2.x and 3.x)
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# 3. Define hyperparameter grid and GridSearchCV
param_grid = {
    "classifier__n_estimators": list((50, 100)),
    "classifier__max_depth": list((None, 10, 20)),
    "classifier__min_samples_split": list((2, 5))
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

# 4. Track experiment, tune hyperparameters, and evaluate with MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Tourism_Package_Prediction")

with mlflow.start_run(run_name="RandomForest_GridSearch"):
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        best_model,
        name="model",
        serialization_format="cloudpickle"
    )

    print("Best Hyperparameters:", grid_search.best_params_)
    print("\n--- Test Set Evaluation ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# 5. Save the best model into tourism_project/deployment/
deployment_dir = "tourism_project/deployment"
os.makedirs(deployment_dir, exist_ok=True)
model_path = os.path.join(deployment_dir, "best_model.joblib")

joblib.dump(best_model, model_path)
print(f"\nBest model saved successfully to: {model_path}")
