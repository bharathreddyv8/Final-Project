"""
Hybrid Model Logic for Fraud Detection
Combines XGBoost (supervised) + Isolation Forest (unsupervised)
Author: Bharath Kumar
"""

import numpy as np


class HybridFraudDetector:
    """
    Hybrid fraud detection combining:
    1. XGBoost - supervised fraud probability
    2. Isolation Forest - anomaly detection score
    
    Final decision based on weighted combination
    """
    
    def __init__(self, xgb_model, iso_model, xgb_weight=0.7, iso_weight=0.3, threshold=0.5):
        """
        Initialize hybrid detector
        
        Args:
            xgb_model: Trained XGBoost model
            iso_model: Trained Isolation Forest model
            xgb_weight: Weight for XGBoost prediction (default: 0.7)
            iso_weight: Weight for Isolation Forest score (default: 0.3)
            threshold: Decision threshold (default: 0.5)
        """
        self.xgb_model = xgb_model
        self.iso_model = iso_model
        self.xgb_weight = xgb_weight
        self.iso_weight = iso_weight
        self.threshold = threshold
        
    def predict_xgboost_batch(self, X_batch):
        """
        Get XGBoost fraud probabilities for batch
        
        Args:
            X_batch: Batch of preprocessed feature arrays
            
        Returns:
            probabilities: Array of fraud probabilities (0-1)
        """
        proba = self.xgb_model.predict_proba(X_batch)[:, 1]
        return proba
    
    def predict_isolation_forest_batch(self, X_batch):
        """
        Get Isolation Forest anomaly scores for batch
        
        Args:
            X_batch: Batch of preprocessed feature arrays
            
        Returns:
            anomaly_scores: Array of normalized anomaly scores (0-1)
        """
        # Get raw anomaly scores
        raw_scores = self.iso_model.score_samples(X_batch)
        
        # Convert to 0-1 scale using sigmoid-like transformation
        anomaly_scores = 1 / (1 + np.exp(raw_scores * 10))
        
        return anomaly_scores
    
    def predict_batch(self, X_batch):
        """
        Make hybrid predictions for batch
        
        Args:
            X_batch: Batch of preprocessed feature arrays
            
        Returns:
            list of dicts with prediction details
        """
        # Get individual model predictions
        xgb_probs = self.predict_xgboost_batch(X_batch)
        iso_scores = self.predict_isolation_forest_batch(X_batch)
        
        # Compute hybrid scores
        hybrid_scores = self.compute_hybrid_score(xgb_probs, iso_scores)
        
        results = []
        for i in range(len(hybrid_scores)):
            # Make final decision
            prediction, confidence = self.make_decision(hybrid_scores[i])
            
            # Prepare result
            result = {
                'xgboost_probability': float(xgb_probs[i]),
                'isolation_forest_score': float(iso_scores[i]),
                'hybrid_risk_score': float(hybrid_scores[i]),
                'final_prediction': prediction,
                'confidence': float(confidence),
                'risk_level': self._get_risk_level(hybrid_scores[i])
            }
            results.append(result)
        
        return results
    
    def predict_xgboost(self, X):
        """
        Get XGBoost fraud probability
        
        Returns:
            probability: Fraud probability (0-1)
        """
        proba = self.xgb_model.predict_proba(X)[:, 1]
        return proba[0]
    
    def predict_isolation_forest(self, X):
        """
        Get Isolation Forest anomaly score
        
        Isolation Forest returns:
        - Negative scores for anomalies
        - Positive scores for normal instances
        
        We convert to 0-1 scale where 1 = anomaly
        
        Returns:
            anomaly_score: Normalized anomaly score (0-1)
        """
        # Get raw anomaly score
        raw_score = self.iso_model.score_samples(X)[0]
        
        # Decision function: -1 for anomalies, 1 for normal
        decision = self.iso_model.predict(X)[0]
        
        # Convert to 0-1 scale
        # More negative score = more anomalous = higher fraud score
        # Typical range is [-0.5, 0.5], but we normalize
        
        # Normalize using sigmoid-like transformation
        anomaly_score = 1 / (1 + np.exp(raw_score * 10))  # Scale raw score
        
        return anomaly_score
    
    def compute_hybrid_score(self, xgb_prob, iso_score):
        """
        Compute weighted hybrid risk score
        
        Args:
            xgb_prob: XGBoost fraud probability
            iso_score: Isolation Forest anomaly score
            
        Returns:
            hybrid_score: Combined risk score (0-1)
        """
        hybrid_score = (self.xgb_weight * xgb_prob) + (self.iso_weight * iso_score)
        return hybrid_score
    
    def make_decision(self, hybrid_score):
        """
        Make final fraud decision based on hybrid score
        
        Args:
            hybrid_score: Combined risk score
            
        Returns:
            prediction: "FRAUD" or "GENUINE"
            confidence: Confidence level
        """
        if hybrid_score >= self.threshold:
            prediction = "FRAUD"
            confidence = hybrid_score
        else:
            prediction = "GENUINE"
            confidence = 1 - hybrid_score
            
        return prediction, confidence
    
    def predict(self, X):
        """
        Make hybrid prediction
        
        Args:
            X: Preprocessed feature array
            
        Returns:
            dict with all prediction details
        """
        # Get individual model predictions
        xgb_prob = self.predict_xgboost(X)
        iso_score = self.predict_isolation_forest(X)
        
        # Compute hybrid score
        hybrid_score = self.compute_hybrid_score(xgb_prob, iso_score)
        
        # Make final decision
        prediction, confidence = self.make_decision(hybrid_score)
        
        # Prepare result
        result = {
            'xgboost_probability': float(xgb_prob),
            'isolation_forest_score': float(iso_score),
            'hybrid_risk_score': float(hybrid_score),
            'final_prediction': prediction,
            'confidence': float(confidence),
            'risk_level': self._get_risk_level(hybrid_score)
        }
        
        return result
    
    def _get_risk_level(self, score):
        """
        Categorize risk level based on score
        
        Args:
            score: Risk score (0-1)
            
        Returns:
            risk_level: LOW, MEDIUM, HIGH, CRITICAL
        """
        if score < 0.3:
            return "LOW"
        elif score < 0.5:
            return "MEDIUM"
        elif score < 0.75:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def get_model_weights(self):
        """
        Return current model weights
        """
        return {
            'xgboost_weight': self.xgb_weight,
            'isolation_forest_weight': self.iso_weight,
            'decision_threshold': self.threshold
        }
