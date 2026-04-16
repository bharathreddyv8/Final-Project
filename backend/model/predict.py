"""
Prediction Module - Load models and make predictions
Author: Bharath Kumar
"""

import joblib
import json
import os
import numpy as np
import pandas as pd
from model.hybrid_logic import HybridFraudDetector
from model.shap_explainer import FraudExplainer


class FraudPredictor:
    """
    Main prediction class that loads models and makes predictions
    """
    
    def __init__(self):
        """
        Load all trained models and artifacts
        """
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        # Load models
        print("Loading models...")
        self.xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.joblib'))
        self.iso_model = joblib.load(os.path.join(models_dir, 'isolation_forest_model.joblib'))
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        
        # Load feature names
        with open(os.path.join(models_dir, 'feature_names.json'), 'r') as f:
            self.feature_names = json.load(f)
        
        # Initialize hybrid detector
        self.hybrid_detector = HybridFraudDetector(
            xgb_model=self.xgb_model,
            iso_model=self.iso_model,
            xgb_weight=0.7,
            iso_weight=0.3,
            threshold=0.5
        )
        
        # Initialize explainer
        self.explainer = FraudExplainer(
            xgb_model=self.xgb_model,
            feature_names=self.feature_names
        )
        
        print("Models loaded successfully!")
        
    def preprocess_input(self, claim_data):
        """
        Preprocess raw claim data for model input
        
        Args:
            claim_data: Dictionary with claim information
            
        Returns:
            X: Preprocessed feature array
        """
        # Extract features in correct order
        features = []
        for feature_name in self.feature_names:
            if feature_name in claim_data:
                features.append(claim_data[feature_name])
            else:
                raise ValueError(f"Missing required feature: {feature_name}")
        
        # Convert to numpy array
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, claim_data):
        """
        Make fraud prediction on a claim
        
        Args:
            claim_data: Dictionary with claim information
            
        Returns:
            result: Dictionary with prediction and explanation
        """
        # Preprocess input
        X = self.preprocess_input(claim_data)
        
        # Make hybrid prediction
        prediction_result = self.hybrid_detector.predict(X)
        
        # Get SHAP explanation
        explanations = self.explainer.explain_prediction(X, top_n=5)
        
        # Generate summary
        summary = self.explainer.generate_explanation_summary(
            explanations, 
            prediction_result['final_prediction']
        )
        
        # Combine results
        result = {
            **prediction_result,
            'explanation': explanations,
            'summary': summary
        }
        
        return result
    
    def predict_batch(self, claims_data):
        """
        Make fraud predictions on multiple claims
        
        Args:
            claims_data: List of dictionaries with claim information
            
        Returns:
            results: List of dictionaries with predictions and explanations
        """
        if not claims_data:
            return []
        
        # Preprocess all inputs
        X_batch = []
        valid_claims = []
        invalid_indices = []
        
        for i, claim_data in enumerate(claims_data):
            try:
                X = self.preprocess_input(claim_data)
                X_batch.append(X.flatten())  # Flatten to 1D for batch processing
                valid_claims.append(claim_data)
            except Exception as e:
                invalid_indices.append((i, str(e)))
        
        if not X_batch:
            raise ValueError("No valid claims to process")
        
        # Convert to numpy array
        X_batch = np.array(X_batch)
        
        # Make hybrid predictions in batch
        batch_results = self.hybrid_detector.predict_batch(X_batch)
        
        # Get SHAP explanations for each prediction
        explanations_batch = []
        summaries_batch = []
        
        for i, X in enumerate(X_batch):
            X_reshaped = X.reshape(1, -1)
            explanations = self.explainer.explain_prediction(X_reshaped, top_n=5)
            explanations_batch.append(explanations)
            
            prediction = batch_results[i]['final_prediction']
            summary = self.explainer.generate_explanation_summary(explanations, prediction)
            summaries_batch.append(summary)
        
        # Combine results
        results = []
        valid_idx = 0
        for i in range(len(claims_data)):
            if any(idx == i for idx, _ in invalid_indices):
                # Find the error for this index
                error = next(error for idx, error in invalid_indices if idx == i)
                results.append({
                    'success': False,
                    'error': error,
                    'claim_index': i
                })
            else:
                result = {
                    **batch_results[valid_idx],
                    'explanation': explanations_batch[valid_idx],
                    'summary': summaries_batch[valid_idx],
                    'success': True,
                    'claim_index': i
                }
                results.append(result)
                valid_idx += 1
        
        return results
    
    def validate_input(self, claim_data):
        """
        Validate input data
        
        Args:
            claim_data: Dictionary with claim information
            
        Returns:
            is_valid: Boolean
            errors: List of validation errors
        """
        errors = []
        
        # Check required fields
        for feature_name in self.feature_names:
            if feature_name not in claim_data:
                errors.append(f"Missing required field: {feature_name}")
        
        # Validate data types and ranges
        if 'age' in claim_data:
            if not (18 <= claim_data['age'] <= 120):
                errors.append("Age must be between 18 and 120")
        
        if 'claim_amount' in claim_data:
            if claim_data['claim_amount'] <= 0:
                errors.append("Claim amount must be positive")
        
        if 'hospital_stay_days' in claim_data:
            if claim_data['hospital_stay_days'] < 0:
                errors.append("Hospital stay cannot be negative")
        
        if 'previous_claims' in claim_data:
            if claim_data['previous_claims'] < 0:
                errors.append("Previous claims cannot be negative")

        if 'policy_age_days' in claim_data:
            if claim_data['policy_age_days'] < 1:
                errors.append("Policy age must be at least 1 day")
        
        is_valid = len(errors) == 0
        
        return is_valid, errors


def get_predictor():
    """
    Singleton pattern to load models once
    """
    if not hasattr(get_predictor, "predictor"):
        get_predictor.predictor = FraudPredictor()
    return get_predictor.predictor
