
# for data manipulation
import pandas as pd
import numpy as np
# for model training, tuning, and evaluation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-prediction-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/indianakhil/tourism-package-prediction-data/Xtrain.csv"
Xtest_path = "hf://datasets/indianakhil/tourism-package-prediction-data/Xtest.csv"
ytrain_path = "hf://datasets/indianakhil/tourism-package-prediction-data/ytrain.csv"
ytest_path = "hf://datasets/indianakhil/tourism-package-prediction-data/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Train multiple models with MLflow tracking
models = {
    'Decision_Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42),
    'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = {}
best_f1_score = 0
best_model_name = None
best_model = None

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(Xtrain, ytrain.values.ravel())

        # Make predictions
        y_pred = model.predict(Xtest)
        y_pred_proba = model.predict_proba(Xtest)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(ytest, y_pred),
            'precision': precision_score(ytest, y_pred, average='weighted'),
            'recall': recall_score(ytest, y_pred, average='weighted'),
            'f1_score': f1_score(ytest, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(ytest, y_pred_proba)
        }

        # Log parameters and metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        results[model_name] = {'model': model, 'metrics': metrics}
        print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")

        # Track best model based on F1-score
        if metrics['f1_score'] > best_f1_score:
            best_f1_score = metrics['f1_score']
            best_model_name = model_name
            best_model = model

print(f"\nBest Model: {best_model_name} with F1-Score: {best_f1_score:.4f}")

# Save the best model locally
model_path = "best_tourism_model_v1.joblib"
joblib.dump(best_model, model_path)

# Log the model artifact
print(f"Model saved as artifact at: {model_path}")

# Upload to Hugging Face
repo_id = "indianakhil/tourism-package-prediction-model"
repo_type = "model"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_file(
    path_or_fileobj="best_tourism_model_v1.joblib",
    path_in_repo="best_tourism_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
