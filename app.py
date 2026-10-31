
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification")
st.write("Machine Learning Model Evaluation Dashboard")

st.markdown(
    "Compare five classification algorithms "
    "using the Dry Bean Dataset."
)

# Load saved models
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load(
            "model/logistic_regression.joblib"
        ),
        "Decision Tree": joblib.load(
            "model/decision_tree.joblib"
        ),
        "KNN": joblib.load(
            "model/knn.joblib"
        ),
        "Naive Bayes": joblib.load(
            "model/naive_bayes.joblib"
        ),
        "Support Vector Machine": joblib.load(
            "model/svm.joblib"
        )
    }

    scaler = joblib.load("model/scaler.joblib")
    return models, scaler

models, scaler = load_models()

# Upload CSV
st.subheader("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.write("Dataset Preview")
    st.dataframe(data.head())

    # Check target column
    if "Class" not in data.columns:
        st.error("Target column 'Class' is missing.")
        st.stop()

    X_test = data.drop(columns=["Class"])
    y_test = data["Class"]

    # Select model
    st.subheader("2. Select Classification Model")

    selected_model = st.selectbox(
        "Choose a model",
        list(models.keys())
    )

    model = models[selected_model]

    # Apply scaling where required
    if selected_model in [
        "Logistic Regression",
        "KNN",
        "Support Vector Machine"
    ]:
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test

    # Predictions using saved model
    y_pred = model.predict(X_input)
    y_prob = model.predict_proba(X_input)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="weighted",
        labels=model.classes_
    )

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

    # Display metrics
    st.subheader("3. Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # Confusion Matrix
    st.subheader("4. Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(selected_model + " - Confusion Matrix")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Classification Report
    st.subheader("5. Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(4)
    )

else:
    st.info("Please upload test_data.csv to begin.")
