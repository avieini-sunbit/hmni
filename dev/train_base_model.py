import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    """Load the prepared data splits."""
    print("Loading data splits...")
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    y_test = pd.read_csv('y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def train_base_model():
    """Train and evaluate the Random Forest model."""
    print("\n=== Phase 2: Training Base Model (Random Forest) ===")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Create pipeline
    pipeline = make_pipeline(
        MaxAbsScaler(),
        MinMaxScaler(),
        RandomForestClassifier(
            bootstrap=False,
            criterion="gini",
            max_features=0.25,
            min_samples_leaf=1,
            min_samples_split=4,
            n_estimators=100
        )
    )
    
    # Remove non-feature columns
    feature_cols = [col for col in X_train.columns if col not in ['a', 'b', 'name_a', 'name_b']]
    X_train_feat = X_train[feature_cols]
    X_test_feat = X_test[feature_cols]
    
    # Train model
    print("Training model...")
    pipeline.fit(X_train_feat, y_train)
    
    # Evaluate
    print("Evaluating model...")
    predictions = pipeline.predict_proba(X_test_feat)
    y_pred_class = [1 if p[1] >= 0.5 else 0 for p in predictions]
    
    print("\nBase Model Evaluation:")
    print(classification_report(y_test, y_pred_class))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_class))
    
    # Save model
    print("\nSaving model...")
    joblib.dump(pipeline, filename='../hmni/models/latin/base_model.pkl')
    
    return pipeline

if __name__ == "__main__":
    train_base_model()

