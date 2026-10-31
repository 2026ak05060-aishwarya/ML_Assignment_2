# Dry Bean Classification Using Machine Learning

## 1. Project Overview

This project focuses on classifying seven varieties of dry beans based on their geometrical and shape-related features using supervised machine learning algorithms.

Five classification algorithms were implemented and evaluated to compare their performance. A Streamlit web application was also developed to allow users to upload test data, select a trained model, and visualize its classification performance.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The dataset contains 13,611 dry bean samples, each represented by 16 numerical features extracted from bean images.

| Property | Details |
|---|---|
| Original instances | 13,611 |
| Number of input features | 16 |
| Target column | Class |
| Number of classes | 7 |
| Classification type | Multiclass |
| Missing values | 0 |
| Duplicate records | 68 |
| Instances after cleaning | 13,543 |

The seven bean classes are **Seker, Barbunya, Bombay, Cali, Dermason, Horoz, and Sira**.

The features include Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, and ShapeFactor4.

## 3. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset from an Excel file using Pandas.
2. Examined the dataset structure and statistical summary.
3. Checked for missing values and duplicate records.
4. Removed 68 duplicate records.
5. Separated the 16 numerical features from the target variable `Class`.
6. Split the cleaned dataset into 80% training and 20% testing data using stratified sampling.
7. Applied StandardScaler to Logistic Regression, KNN, and SVM.
8. Used unscaled features for Decision Tree and Gaussian Naive Bayes.
9. Saved the test dataset as `test_data.csv` for Streamlit evaluation.

The dataset was divided into **10,834 training samples** and **2,709 testing samples**.

## 4. Machine Learning Models

Five supervised machine learning algorithms were implemented using Scikit-learn.

### 4.1 Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of samples belonging to different classes.

### 4.2 Decision Tree

Decision Tree classifies samples by recursively dividing the dataset based on feature values and decision rules.

### 4.3 K-Nearest Neighbors (KNN)

KNN classifies a sample based on the majority class among its nearest neighbors. The model was implemented using `k = 5`.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem. It assumes that numerical features follow Gaussian distributions within each class and are conditionally independent.

### 4.5 Support Vector Machine (SVM)

SVM identifies decision boundaries between classes. An RBF kernel was used to support nonlinear classification.

## 5. Model Evaluation

The models were evaluated using six performance metrics:

- **Accuracy:** Proportion of correctly classified samples.
- **AUC:** Weighted multiclass One-vs-Rest ROC AUC.
- **Precision:** Weighted average precision across classes.
- **Recall:** Weighted average recall across classes.
- **F1 Score:** Weighted harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient, which measures overall classification quality.

### 5.1 Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | **0.9934** | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.8966 | 0.9363 | 0.8965 | 0.8966 | 0.8964 | 0.8750 |
| K-Nearest Neighbors | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| **Support Vector Machine** | **0.9210** | 0.9933 | **0.9214** | **0.9210** | **0.9211** | **0.9045** |

## 6. Model Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 91.92% and an F1 score of 0.9193. It recorded the highest AUC of 0.9934, indicating strong class discrimination. Its overall performance was close to that of SVM.

### Decision Tree

Decision Tree achieved an accuracy of 89.66% and an F1 score of 0.8964. Its performance was lower than Logistic Regression, KNN, and SVM, although it still provided reasonably accurate classification results.

### K-Nearest Neighbors

KNN achieved an accuracy of 91.55% and an F1 score of 0.9157. It demonstrated competitive classification performance after feature scaling, with results close to Logistic Regression and SVM.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved the lowest accuracy of 76.30% and an F1 score of 0.7607. However, its AUC remained relatively high at 0.9644. Its distributional and feature-independence assumptions may have limited its classification performance.

### Support Vector Machine

SVM achieved the highest accuracy of 92.10%, F1 score of 0.9211, and MCC of 0.9045. It also recorded a high AUC of 0.9933, demonstrating strong overall classification performance.

## 7. Best Performing Model

**Selected Model: Support Vector Machine (SVM)**

SVM was identified as the best-performing model because it achieved the highest Accuracy, Precision, Recall, F1 Score, and MCC among the five algorithms.

Although Logistic Regression achieved a marginally higher AUC, SVM demonstrated the strongest overall performance on the test dataset.

## 8. Streamlit Web Application

A Streamlit application was developed to evaluate the saved classification models through an interactive interface.

### Application Features

- Upload a CSV test dataset containing all 16 input features and the `Class` target column.
- Select one of the five trained classification models.
- Generate predictions using the selected model.
- Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Visualize the confusion matrix.
- Display a detailed classification report.

The application loads previously trained models and preprocessing artifacts without retraining.

## 9. Project Structure

```text
Dry-Bean-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
├── notebook.ipynb
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── scaler.joblib
```

## 10. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Google Colab
- GitHub

## 11. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Dry-Bean-Classification.git
cd Dry-Bean-Classification
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit application

```bash
streamlit run app.py
```

### Step 4: Evaluate the models

1. Open the Streamlit application.
2. Upload `test_data.csv`.
3. Select a classification model.
4. View the evaluation metrics, confusion matrix, and classification report.

## 12. Project Links

**GitHub Repository:** [Add your GitHub repository URL]

**Streamlit Application:** [Add your deployed Streamlit URL]

## 13. Conclusion

Five supervised machine learning algorithms were implemented and compared for multiclass classification of dry beans.

Support Vector Machine achieved the best overall performance with **92.10% accuracy**, followed by Logistic Regression with 91.92% and KNN with 91.55%.

Gaussian Naive Bayes showed the lowest classification accuracy among the evaluated models.

The results highlight the importance of comparing multiple machine learning algorithms using different evaluation metrics. The Streamlit application provides a convenient interface for evaluating saved models and visualizing their predictions.

## 14. Reference

Koklu, M., and Ozkan, I. A. (2020). *Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques.* Computers and Electronics in Agriculture, 174, 105507.

UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
