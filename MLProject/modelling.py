import pandas as pd
import numpy as np
import os
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description="Retrain Random Forest model with arguments")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in forest")
    parser.add_argument("--max_depth", type=int, default=8, help="Maximum depth of tree")
    return parser.parse_args()

def retrain_model():
    args = parse_args()
    
    # Enable MLflow Autologging
    mlflow.sklearn.autolog()
    
    # 1. Load data
    data_path = "./diabetes_preprocessed.csv"
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # 2. Split train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train Model
    print(f"Training Random Forest Classifier (n_estimators={args.n_estimators}, max_depth={args.max_depth}) with autologging...")
    with mlflow.start_run(run_name=f"RandomForest_Retrain_n_{args.n_estimators}_d_{args.max_depth}"):
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict & Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    retrain_model()
