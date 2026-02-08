
# for data manipulation
import pandas as pd
import numpy as np
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/indianakhil/tourism-package-prediction-data/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Remove unnecessary columns
if 'Unnamed: 0' in tourism_dataset.columns:
    tourism_dataset = tourism_dataset.drop('Unnamed: 0', axis=1)
if 'CustomerID' in tourism_dataset.columns:
    tourism_dataset = tourism_dataset.drop('CustomerID', axis=1)

# Handle missing values
# Fill numerical missing values with median
numerical_cols = tourism_dataset.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if tourism_dataset[col].isnull().sum() > 0:
        tourism_dataset[col].fillna(tourism_dataset[col].median(), inplace=True)

# Fill categorical missing values with mode
categorical_cols = tourism_dataset.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if tourism_dataset[col].isnull().sum() > 0:
        tourism_dataset[col].fillna(tourism_dataset[col].mode()[0], inplace=True)

# Encode categorical features using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    tourism_dataset[col] = le.fit_transform(tourism_dataset[col].astype(str))
    label_encoders[col] = le

# Define the target variable for the classification task
target = 'ProdTaken'

# Define predictor matrix (X) and target variable (y)
X = tourism_dataset.drop(target, axis=1)
y = tourism_dataset[target]

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,   # Ensures reproducibility by setting a fixed random seed
    stratify=y         # Stratify to maintain class distribution
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="indianakhil/tourism-package-prediction-data",
        repo_type="dataset",
    )
