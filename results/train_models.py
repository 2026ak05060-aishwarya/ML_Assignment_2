
"""
Dry Bean Classification
Data preprocessing, model training and evaluation.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

DATA_FILE = DATA_DIR / "Dry_Bean_Dataset.xlsx"

# Create directories if they do not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading Dry Bean dataset...")

data = pd.read_excel(DATA_FILE)

print("Original dataset shape:", data.shape)


# --------------------------------------------------
# 3. DATA PREPROCESSING
# --------------------------------------------------

# Remove duplicate records
data = data.drop_duplicates()

print("Shape after removing duplicates:", data.shape)

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

FEATURES = X.columns.tolist()
CLASS_NAMES = sorted(y.unique().tolist())

print("Number of features:", len(FEATURES))
print("Number of classes:", len(CLASS_NAMES))


# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 5. FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save fitted scaler
joblib.dump(
    scaler,
    MODEL_DIR / "scaler.joblib",
    compress=3
)

print("\nScaler saved successfully.")


# --------------------------------------------------
# 6. SAVE TEST DATASETS
# --------------------------------------------------

# Test dataset with actual class labels
test_data = X_test.copy()
test_data["Class"] = y_test

test_data.to_csv(
    BASE_DIR / "test_data.csv",
    index=False
)

# Test dataset without class labels
X_test.to_csv(
    BASE_DIR / "test_data_without_labels.csv",
    index=False
)

print("Test datasets saved successfully.")


# --------------------------------------------------
# 7. DEFINE CLASSIFICATION MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        "filename": "logistic_regression.joblib",
        "scaled": True
    },

    "Decision Tree": {
        "model": DecisionTreeClassifier(
            random_state=42
        ),
        "filename": "decision_tree.joblib",
        "scaled": False
    },

    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(
            n_neighbors=5
        ),
        "filename": "knn.joblib",
        "scaled": True
    },

    "Gaussian Naive Bayes": {
        "model": GaussianNB(),
        "filename": "naive_bayes.joblib",
        "scaled": False
    },

    "Support Vector Machine": {
        "model": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),
        "filename": "svm.joblib",
        "scaled": True
    }
}


# --------------------------------------------------
# 8. TRAIN MODELS AND SAVE JOBLIB FILES
# --------------------------------------------------

results = []

print("\nStarting model training...")


for name, details in models.items():

    print(f"\nTraining {name}...")

    model = details["model"]

    # Select appropriate training and testing data
    if details["scaled"]:
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    else:
        X_train_model = X_train
        X_test_model = X_test

    # Train model
    model.fit(X_train_model, y_train)

    # Generate predictions
    y_pred = model.predict(X_test_model)

    # Calculate probabilities for multiclass AUC
    y_prob = model.predict_proba(X_test_model)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="weighted",
        labels=model.classes_
    )

    # Save trained model
    model_path = MODEL_DIR / details["filename"]

    joblib.dump(
        model,
        model_path,
        compress=3
    )

    # Verify saved model
    loaded_model = joblib.load(model_path)

    assert list(loaded_model.classes_) == list(model.classes_)

    print("Model trained and saved successfully.")
    print("File:", model_path.name)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })


# --------------------------------------------------
# 9. SAVE MODEL COMPARISON RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

results_path = RESULTS_DIR / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\nModel comparison results:")
print(results_df.to_string(index=False))


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("DRY BEAN CLASSIFICATION - TRAINING COMPLETED")
print("=" * 50)

print("\nGenerated model files:")

for file in MODEL_DIR.glob("*.joblib"):
    print(file.name, "-", round(file.stat().st_size / 1024, 2), "KB")

print("\nGenerated result files:")
print("model_comparison.csv")
print("test_data.csv")
print("test_data_without_labels.csv")

print("\nAll models trained and saved successfully!")
