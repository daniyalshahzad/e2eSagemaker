import json
import pathlib
import pickle
import tarfile
import argparse
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss, average_precision_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score


import logging

logging.basicConfig(level=logging.DEBUG)


def calculate_ece(prob_true, prob_pred, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True labels.
    - y_prob: array-like of shape (n_samples,)
        Predicted probabilities.
    - n_bins: int, default=10
        Number of bins to use for calibration curve.

    Returns:
    - ece: float
        Expected Calibration Error.
    """
    bin_totals = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    ece = np.sum(bin_totals * np.abs(prob_true - prob_pred)) / np.sum(bin_totals)
    return ece

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--test-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()
    
    logging.info("Starting evaluation script...")

    # Extract model.tar.gz
    model_tar_path = os.path.join(args.model_path, 'model.tar.gz')
    with tarfile.open(model_tar_path) as tar:
        tar.extractall(path=".")
                      
    logging.info("Model tar extracted successfully.")

    # Load the model
    model = pickle.load(open("xgboost-model", "rb"))
                      
    logging.info("Model loaded successfully.")

    # Load the test data
    test_path = os.path.join(args.test_path, 'processed_test.csv')
    df = pd.read_csv(test_path, header=None)
    
    train = pd.read_csv(os.path.join(args.train_path, 'processed_train.csv'), header=None)
    test = pd.read_csv(os.path.join(args.test_path, 'processed_test.csv'), header=None)
    
    X_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values

    X_test = test.iloc[:, 1:].values
    y_test = test.iloc[:, 0].values

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    baseline_probabilistic_prediction = y_train.mean()
    baseline_predictions = np.full((X_test.shape[0], 1), baseline_probabilistic_prediction)

    # Make predictions
    predictions = model.predict(dtest) #This is probabilistic predictions
    train_predictions = model.predict(dtrain) 

    # Compute PR AUC using average_precision_score for the model (both training and test sets)
    model_pr_auc_train = average_precision_score(y_train, train_predictions)
    model_pr_auc_test = average_precision_score(y_test, predictions)

    # The baseline PR AUC is simply the positive class rate (i.e., y_train.mean())
    baseline_pr_auc_test = y_train.mean()  # Baseline for test set uses training positive class rate

    # Print the PR AUC values
    print(f'Model Train PR AUC: {model_pr_auc_train}')
    print(f'Model Test PR AUC: {model_pr_auc_test}')
    print(f'Baseline PR AUC: {baseline_pr_auc_test}')

    # Compute PR AUC Skill Score (PRSS) for both training and test sets
    PRSS_test = (model_pr_auc_test - baseline_pr_auc_test) / (1 - baseline_pr_auc_test)
    print(f'Test PR AUC Skill Score (PRSS): {PRSS_test}')
    print('Improvement over baseline: ', (model_pr_auc_test - baseline_pr_auc_test) / baseline_pr_auc_test)
    print('-----')

    train_brier_score = brier_score_loss(y_train, train_predictions)
    test_brier_score = brier_score_loss(y_test, predictions)

    baseline_brier_score =  brier_score_loss(y_test, baseline_predictions)
    BSS = 1 - (test_brier_score / baseline_brier_score)

    print('Train Brier Score: ', train_brier_score)
    print('Test Brier Score: ', test_brier_score)
    print('Baseline Brier Score: ', baseline_brier_score)
    print('Brier Skill Score: ', BSS)


    # Prepare evaluation report
    report_dict = {
        "classification_metrics": {
            "BSS": {"value": BSS},
            "PR-AUC" : {"value": model_pr_auc_test},
            "PR-AUC Improvement" : {"value": (model_pr_auc_test - baseline_pr_auc_test) / baseline_pr_auc_test}
        }
    }

    # Save the evaluation report
    output_dir = args.output_path
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = os.path.join(output_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    logging.info("Evaluation report saved successfully.")