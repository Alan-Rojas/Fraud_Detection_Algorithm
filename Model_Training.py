# Dependencies used troughout the model's training.

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from EDA import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay, precision_score, recall_score, f1_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# End of dependencies´ list

# Workflow
def pipeline(threshold=0.9):
    train_data = get_training_data()
    test_data_x, test_data_y = get_all_data()

    # Train first model on balanced data
    first_model = LogisticRegression(class_weight="balanced")
    x_train = train_data.drop(columns="Class")
    y_train = train_data["Class"]
    first_model.fit(x_train, y_train)

    # Predict on the full (imbalanced) dataset
    probs = first_model.predict_proba(test_data_x)[:, 1]
    first_predictions = (probs > threshold).astype(int)

    # Identify suspicious transactions (predicted as fraud)
    suspicious_x = test_data_x[first_predictions == 1]
    suspicious_y = test_data_y[first_predictions == 1]  # True labels of suspicious

    rndm_x_train, rndm_x_test, rndm_y_train, rndm_y_test = train_test_split(
        suspicious_x, suspicious_y, train_size=0.8, random_state=42
    )

    # Train second model on suspicious cases
    second_model = RandomForestClassifier(n_estimators=30, random_state=42)
    second_model.fit(rndm_x_train, rndm_y_train)
    second_predictions = second_model.predict(rndm_x_test)

    return {
        "logistic_model": first_model,
        "random_forest_model": second_model,
        "threshold": threshold,

        "full_test_x": test_data_x,
        "full_test_y": test_data_y,
        "logistic_probs": probs,
        "logistic_predictions": first_predictions,

        "suspicious_x_train": rndm_x_train,
        "suspicious_x_test": rndm_x_test,
        "suspicious_y_train": rndm_y_train,
        "suspicious_y_test": rndm_y_test,
        "rf_predictions": second_predictions
    }

def plot_pipeline_metrics(precision_vals, recall_vals):
    """
    Generate precision and recall bar plots across pipeline stages.
    Returns a matplotlib figure for use in Streamlit.
    
    Parameters:
    - precision_vals: list or tuple with precision values [baseline, stage1, stage2]
    - recall_vals: list or tuple with recall values [baseline, stage1, stage2]
    
    Returns:
    - fig: matplotlib figure object
    """
    stages = ["Before Pipeline", "Stage 1 (LogReg)", "Stage 2 (RandForest)"]
    colors = ["gray", "skyblue", "orange"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Precision
    sns.barplot(x=stages, y=precision_vals, palette=colors, ax=axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("% Predicted Fraud are Actual Fraud")
    axes[0].set_ylabel("Precision")
    for i, val in enumerate(precision_vals):
        axes[0].text(i, val + 0.02, f"{val*100:.2f}%", ha='center', fontweight='bold')

    # Recall
    sns.barplot(x=stages, y=recall_vals, palette=colors, ax=axes[1])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("% of Fraud Caught")
    axes[1].set_ylabel("Recall")
    for i, val in enumerate(recall_vals):
        axes[1].text(i, val + 0.02, f"{val*100:.2f}%", ha='center', fontweight='bold')

    fig.tight_layout()
    return fig

def analyze_pipeline_results(y_true_all, y_pred_all, y_true_suspicious, y_pred_suspicious):
    results = {}

    # First stage metrics
    cm1 = confusion_matrix(y_true_all, y_pred_all)
    stage_1_res = {
        "Recall": recall_score(y_true_all, y_pred_all),
        "Precision": precision_score(y_true_all, y_pred_all),
        "F1 Score": f1_score(y_true_all, y_pred_all),
        "Accuracy": accuracy_score(y_true_all, y_pred_all)
    }

    # First stage confusion matrix plot
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("Confusion Matrix - Logistic Regression")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    fig1.tight_layout()

    # Second stage metrics
    cm2 = confusion_matrix(y_true_suspicious, y_pred_suspicious)
    stage_2_res = {
        "Recall": recall_score(y_true_suspicious, y_pred_suspicious),
        "Precision": precision_score(y_true_suspicious, y_pred_suspicious),
        "F1 Score": f1_score(y_true_suspicious, y_pred_suspicious),
        "Accuracy": accuracy_score(y_true_suspicious, y_pred_suspicious)
    }

    # Second stage confusion matrix plot
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges", ax=ax2)
    ax2.set_title("Confusion Matrix - Random Forest (Suspicious Cases)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    fig2.tight_layout()

    # Optional: Metrics for overall progress chart
    recall_progress = [0.5, stage_1_res["Recall"], stage_2_res["Recall"]]
    precision_progress = [0.6, stage_1_res["Precision"], stage_2_res["Precision"]]

    results = {
        "stage_1": {
            "metrics": stage_1_res,
            "confusion_matrix_fig": fig1
        },
        "stage_2": {
            "metrics": stage_2_res,
            "confusion_matrix_fig": fig2
        },
        "progress": {
            "recall": recall_progress,
            "precision": precision_progress
        }
    }

    return results

