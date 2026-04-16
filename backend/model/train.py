"""
Medical Insurance Claim Fraud Detection - Training Pipeline
Author: Bharath Kumar
Description: Trains XGBoost and Isolation Forest models for hybrid fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import os
import json

# Set random seed for reproducibility
np.random.seed(42)


def generate_sample_dataset(n_samples=5000):
    """
    Generate BALANCED synthetic medical insurance claim dataset
    50% Fraud, 50% Non-Fraud
    
    Features include:
    - Patient demographics
    - Claim details
    - Provider information
    - Medical codes
    """
    print("Generating BALANCED medical insurance claim dataset (50% Fraud / 50% Non-Fraud)...")
    
    half_samples = n_samples // 2
    
    # Initialize arrays for all samples
    age = np.zeros(n_samples, dtype=int)
    gender = np.zeros(n_samples, dtype=int)
    claim_amount = np.zeros(n_samples, dtype=float)
    hospital_stay = np.zeros(n_samples, dtype=int)
    previous_claims = np.zeros(n_samples, dtype=int)
    treatment_type = np.zeros(n_samples, dtype=int)
    provider_type = np.zeros(n_samples, dtype=int)
    diagnosis_code = np.zeros(n_samples, dtype=int)
    procedure_code = np.zeros(n_samples, dtype=int)
    chronic_condition = np.zeros(n_samples, dtype=int)
    insurance_type = np.zeros(n_samples, dtype=int)
    policy_age = np.zeros(n_samples, dtype=int)
    beneficiaries = np.zeros(n_samples, dtype=int)
    fraud = np.zeros(n_samples, dtype=int)
    
    # Generate LEGITIMATE (Non-Fraud) Claims - First Half
    print(f"  Generating {half_samples} LEGITIMATE claims...")
    for i in range(half_samples):
        age[i] = np.random.randint(25, 85)
        gender[i] = np.random.choice([0, 1])
        
        # More realistic with some legitimate high-cost claims
        claim_type = np.random.choice(['routine', 'moderate', 'serious'], p=[0.65, 0.28, 0.07])
        
        if claim_type == 'routine':
            claim_amount[i] = np.random.uniform(500, 8000)
            hospital_stay[i] = np.random.randint(0, 5)
            previous_claims[i] = np.random.randint(0, 4)
            treatment_type[i] = np.random.choice([0, 1], p=[0.8, 0.2])
        elif claim_type == 'moderate':
            claim_amount[i] = np.random.uniform(8000, 18000)
            hospital_stay[i] = np.random.randint(3, 10)
            previous_claims[i] = np.random.randint(0, 5)
            treatment_type[i] = np.random.choice([1, 2], p=[0.7, 0.3])
        else:  # serious legitimate cases (fewer now)
            claim_amount[i] = np.random.uniform(18000, 38000)  # Some overlap
            hospital_stay[i] = np.random.randint(8, 21)  # Longer stays justify high cost
            previous_claims[i] = np.random.randint(0, 4)
            treatment_type[i] = np.random.choice([1, 2], p=[0.6, 0.4])
        
        provider_type[i] = np.random.choice([0, 1, 2], p=[0.65, 0.28, 0.07])
        diagnosis_code[i] = np.random.randint(100, 999)
        procedure_code[i] = np.random.randint(1000, 9999)
        chronic_condition[i] = np.random.choice([0, 1], p=[0.68, 0.32])
        insurance_type[i] = np.random.choice([0, 1, 2], p=[0.55, 0.32, 0.13])
        policy_age[i] = np.random.randint(120, 2000)  # Established policies
        beneficiaries[i] = np.random.randint(1, 6)
        fraud[i] = 0  # Non-fraud
    
    # Generate FRAUDULENT Claims - Second Half
    print(f"  Generating {half_samples} FRAUDULENT claims...")
    for i in range(half_samples, n_samples):
        age[i] = np.random.randint(20, 88)
        gender[i] = np.random.choice([0, 1])
        
        # FRAUD CHARACTERISTICS - Clear patterns with some edge cases
        fraud_type = np.random.choice(['high_amount', 'suspicious_pattern', 'new_policy_fraud', 'emergency_fraud', 'subtle_fraud', 'edge_case'], 
                                       p=[0.28, 0.28, 0.18, 0.14, 0.08, 0.04])  # Added edge cases
        
        # Less noise to make patterns clearer
        noise_factor = np.random.uniform(0.85, 1.15)
        
        if fraud_type == 'high_amount':
            # High amounts with clear indicators
            claim_amount[i] = np.random.uniform(22000, 50000) * noise_factor
            hospital_stay[i] = np.random.randint(0, 5)  # Short stays
            previous_claims[i] = np.random.randint(0, 6)
            treatment_type[i] = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
            
        elif fraud_type == 'suspicious_pattern':
            # Suspicious patterns - clearer signals
            claim_amount[i] = np.random.uniform(15000, 38000) * noise_factor
            hospital_stay[i] = np.random.randint(0, 3)  # Very short stays
            previous_claims[i] = np.random.randint(7, 16)  # Many claims - clear red flag
            treatment_type[i] = np.random.choice([0, 2], p=[0.3, 0.7])
            
        elif fraud_type == 'new_policy_fraud':
            # New policies with high claims - clearer pattern
            claim_amount[i] = np.random.uniform(20000, 45000) * noise_factor
            hospital_stay[i] = np.random.randint(0, 6)
            previous_claims[i] = np.random.randint(0, 3)
            treatment_type[i] = np.random.choice([1, 2], p=[0.3, 0.7])
            policy_age[i] = np.random.randint(1, 90)  # Very new policies
            
        elif fraud_type == 'emergency_fraud':
            # Emergency treatment fraud with clear signals
            claim_amount[i] = np.random.uniform(16000, 42000) * noise_factor
            hospital_stay[i] = np.random.randint(0, 2)  # Minimal stay
            previous_claims[i] = np.random.randint(5, 13)
            treatment_type[i] = 2  # Emergency
            provider_type[i] = np.random.choice([1, 2], p=[0.4, 0.6])
            
        elif fraud_type == 'subtle_fraud':
            # Looks almost legitimate - causes false negatives
            claim_amount[i] = np.random.uniform(12000, 28000) * noise_factor
            hospital_stay[i] = np.random.randint(3, 9)
            previous_claims[i] = np.random.randint(4, 7)
            treatment_type[i] = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
            provider_type[i] = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            
        else:  # edge_case - borderline cases that can be missed
            # Very tricky cases - looks very legitimate
            claim_amount[i] = np.random.uniform(10000, 22000) * noise_factor
            hospital_stay[i] = np.random.randint(4, 12)  # Reasonable stays
            previous_claims[i] = np.random.randint(2, 5)  # Normal claim history
            treatment_type[i] = np.random.choice([0, 1], p=[0.5, 0.5])
            provider_type[i] = np.random.choice([0, 1], p=[0.6, 0.4])
            chronic_condition[i] = 1  # Has chronic condition
            insurance_type[i] = np.random.choice([0, 1], p=[0.6, 0.4])
        
        # Common fraud patterns with clearer signals
        gender[i] = np.random.choice([0, 1])
        
        # More distinct provider patterns for fraud
        if provider_type[i] == 0:  # If not set by fraud type
            provider_type[i] = np.random.choice([0, 1, 2], p=[0.25, 0.40, 0.35])
        
        diagnosis_code[i] = np.random.randint(100, 999)
        procedure_code[i] = np.random.randint(1000, 9999)
        chronic_condition[i] = np.random.choice([0, 1], p=[0.45, 0.55])  # Slightly more chronic
        insurance_type[i] = np.random.choice([0, 1, 2], p=[0.15, 0.50, 0.35])  # More premium types
        
        if policy_age[i] == 0:  # If not already set
            policy_age[i] = np.random.randint(1, 1600)
        
        beneficiaries[i] = np.random.randint(1, 6)
        fraud[i] = 1  # Fraud
    
    # Clip claim amounts
    claim_amount = np.clip(claim_amount, 500, 50000)
    
    # Create DataFrame (without claim_id - not needed for model)
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'chronic_condition': chronic_condition,
        'beneficiaries': beneficiaries,
        'claim_amount': claim_amount.round(2),
        'hospital_stay_days': hospital_stay,
        'previous_claims': previous_claims,
        'treatment_type': treatment_type,
        'diagnosis_code': diagnosis_code,
        'procedure_code': procedure_code,
        'provider_type': provider_type,
        'insurance_type': insurance_type,
        'policy_age_days': policy_age,
        'is_fraud': fraud
    })
    
    print(f"Dataset generated: {len(df)} samples")
    print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Genuine cases: {(~df['is_fraud'].astype(bool)).sum()} ({(1-df['is_fraud'].mean())*100:.2f}%)")
    
    return df


def preprocess_data(df):
    """
    Preprocess the dataset for model training
    """
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']
    
    # Feature names for later use
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classifier for supervised fraud detection
    """
    print("\n" + "="*60)
    print("Training XGBoost Classifier...")
    print("="*60)
    
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    scale_pos_weight = class_weights[1] / class_weights[0]
    
    print(f"Class weights: Non-fraud={class_weights[0]:.4f}, Fraud={class_weights[1]:.4f}")
    print(f"Scale pos weight: {scale_pos_weight:.4f}")
    
    # XGBoost parameters with class weight adjustment
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'use_label_encoder': False,
        'scale_pos_weight': scale_pos_weight  # Handle class imbalance
    }
    
    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(auc)
    }
    
    return model, metrics


def train_isolation_forest(X_train, y_train=None):
    """
    Train Isolation Forest for anomaly detection
    """
    print("\n" + "="*60)
    print("Training Isolation Forest...")
    print("="*60)
    
    # Calculate actual fraud rate from training data if available
    if y_train is not None:
        actual_fraud_rate = y_train.mean()
        contamination = max(actual_fraud_rate, 0.01)  # Use at least 1% to detect rare anomalies
        print(f"Actual fraud rate: {actual_fraud_rate:.4f} ({actual_fraud_rate*100:.2f}%)")
    else:
        contamination = 0.15  # Default expected fraud rate
    
    print(f"Contamination parameter: {contamination:.4f}")
    
    # Isolation Forest parameters
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto',
        max_features=1.0
    )
    
    # Fit model
    model.fit(X_train)
    
    # Get anomaly scores
    scores = model.score_samples(X_train)
    print(f"Mean anomaly score: {scores.mean():.4f}")
    print(f"Std anomaly score:  {scores.std():.4f}")
    
    return model


def save_models(xgb_model, iso_model, scaler, feature_names, metrics=None):
    """
    Save trained models and preprocessing artifacts
    """
    print("\n" + "="*60)
    print("Saving models...")
    print("="*60)
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save XGBoost model
    xgb_path = os.path.join(models_dir, 'xgboost_model.joblib')
    joblib.dump(xgb_model, xgb_path)
    print(f"✓ XGBoost model saved: {xgb_path}")
    
    # Save Isolation Forest model
    iso_path = os.path.join(models_dir, 'isolation_forest_model.joblib')
    joblib.dump(iso_model, iso_path)
    print(f"✓ Isolation Forest model saved: {iso_path}")
    
    # Save scaler
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(models_dir, 'feature_names.json')
    with open(features_path, 'w') as f:
        json.dump(feature_names, f)
    print(f"✓ Feature names saved: {features_path}")
    
    # Save evaluation metrics
    if metrics:
        metrics_path = os.path.join(models_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"✓ Evaluation metrics saved: {metrics_path}")
    
    print("\nAll models saved successfully!")


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*60)
    print("MEDICAL INSURANCE FRAUD DETECTION - TRAINING PIPELINE")
    print("Student: Bharath Kumar")
    print("="*60)
    
    # Generate dataset
    df = generate_sample_dataset(n_samples=5000)
    
    # Save dataset for reference
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, 'insurance_claims_dataset.csv')
    df.to_csv(dataset_path, index=False)
    print(f"\n✓ Dataset saved: {dataset_path}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Train XGBoost
    xgb_model, metrics = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Train Isolation Forest (pass y_train for actual fraud rate calculation)
    iso_model = train_isolation_forest(X_train, y_train)
    
    # Save models
    save_models(xgb_model, iso_model, scaler, feature_names, metrics)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now run the FastAPI server to make predictions.")


if __name__ == "__main__":
    main()
