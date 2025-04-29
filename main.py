# main.py
import pandas as pd
import numpy as np
import argparse
import os
from model import XGBoostModel
from tuning import tune_xgboost, load_params, save_params
from load_data import load_data, load_senti_data
from feature_importance import analyze_feature_importance
from config import *
from utils import *

def create_output_dirs():
    """Create necessary output directories"""
    dirs = ["analysis_output", "analysis_output/with_sentiment", "analysis_output/without_sentiment"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def main(tune=False, use_sentiment=False):
    # Create output directories
    create_output_dirs()
    
    # Load appropriate dataset
    if use_sentiment:
        X_train, y_train, X_val, y_val = load_senti_data()
        params_file = 'best_params_senti.json'
        feature_names = numerical_features + senti_features
        output_dir = "analysis_output/with_sentiment"
        print("Using data with sentiment features")
    else:
        X_train, y_train, X_val, y_val = load_data()
        params_file = 'best_params.json'
        feature_names = numerical_features
        output_dir = "analysis_output/without_sentiment"
        print("Using data without sentiment features")

    # Hyperparameter tuning or loading
    if tune:
        print("Running hyperparameter tuning...")
        best_model, best_params, best_score, val_score = tune_xgboost(X_train, y_train, X_val, y_val)
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation score: {best_score:.4f}")
        print(f"Validation score: {val_score:.4f}")
        save_params(best_params, params_file)
    else:
        try:
            best_params = load_params(params_file)
            print(f"Loaded parameters from {params_file}")
        except FileNotFoundError:
            print(f"No saved parameters found. Using default XGBoost parameters.")
            best_params = {}

    # Train and evaluate model
    try:
        model = XGBoostModel(params=best_params)
        model.train(X_train, y_train)
        val_metrics = model.evaluate(X_val, y_val)
        print(f"Validation metrics {'with' if use_sentiment else 'without'} sentiment:", 
              {k: f"{v:.4f}" for k, v in val_metrics.items()})

        # Perform feature importance analysis
        print("\nPerforming feature importance analysis...")
        feature_importance = analyze_feature_importance(model.model, X_val, feature_names, output_dir)
        
        print("\nFeature Importance:")
        for feature, importance in feature_importance.items():
            print(f"{feature}: {importance:.4f}")

    except Exception as e:
        print(f"Error during model training or evaluation: {str(e)}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost model with optional tuning and sentiment features")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    parser.add_argument("--sentiment", action="store_true", help="Use sentiment features")
    args = parser.parse_args()

    main(tune=args.tune, use_sentiment=args.sentiment)