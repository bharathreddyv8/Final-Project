"""
SHAP Explainer for Model Interpretability
Explains why a claim is predicted as fraud
Author: Bharath Kumar
"""

import shap
import numpy as np


class FraudExplainer:
    """
    Provides explainability for fraud predictions using SHAP values
    """
    
    def __init__(self, xgb_model, feature_names):
        """
        Initialize SHAP explainer
        
        Args:
            xgb_model: Trained XGBoost model
            feature_names: List of feature names
        """
        self.xgb_model = xgb_model
        self.feature_names = feature_names
        
        # Initialize SHAP explainer for XGBoost
        # Using TreeExplainer for efficiency with tree-based models
        self.explainer = shap.TreeExplainer(self.xgb_model)
        
    def explain_prediction(self, X, top_n=5):
        """
        Generate SHAP-based explanation for a prediction
        
        Args:
            X: Preprocessed feature array (single sample)
            top_n: Number of top features to return
            
        Returns:
            dict: Explanation with top contributing features
        """
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Get SHAP values for the sample
        if len(shap_values.shape) > 1:
            sample_shap_values = shap_values[0]
        else:
            sample_shap_values = shap_values
            
        # Get feature contributions (absolute SHAP values)
        feature_contributions = np.abs(sample_shap_values)
        
        # Get top N features
        top_indices = np.argsort(feature_contributions)[-top_n:][::-1]
        
        # Prepare explanation
        explanations = []
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            shap_value = float(sample_shap_values[idx])
            feature_value = float(X[0][idx])
            impact = "increases" if shap_value > 0 else "decreases"
            
            # Get human-readable feature name
            readable_name = self._get_readable_feature_name(feature_name)
            
            explanations.append({
                'feature': readable_name,
                'value': feature_value,
                'impact': shap_value,
                'impact_direction': impact,
                'absolute_impact': abs(shap_value)
            })
        
        return explanations
    
    def _get_readable_feature_name(self, feature_name):
        """
        Convert feature names to human-readable format
        
        Args:
            feature_name: Technical feature name
            
        Returns:
            readable_name: Human-readable feature name
        """
        name_mapping = {
            'age': 'Patient Age',
            'gender': 'Gender',
            'claim_amount': 'Claim Amount',
            'hospital_stay_days': 'Hospital Stay Duration',
            'previous_claims': 'Number of Previous Claims',
            'treatment_type': 'Treatment Type',
            'provider_type': 'Provider Type',
            'diagnosis_code': 'Diagnosis Code',
            'procedure_code': 'Procedure Code',
            'chronic_condition': 'Chronic Condition',
            'insurance_type': 'Insurance Type',
            'policy_age_days': 'Policy Age (Days)',
            'beneficiaries': 'Number of Beneficiaries'
        }
        
        return name_mapping.get(feature_name, feature_name)
    
    def generate_explanation_summary(self, explanations, prediction):
        """
        Generate human-readable summary of why claim is fraud/genuine
        
        Args:
            explanations: List of feature explanations
            prediction: Final prediction (FRAUD or GENUINE)
            
        Returns:
            summary: Text summary
        """
        if prediction == "FRAUD":
            summary = "This claim is flagged as FRAUD due to the following suspicious factors:\n\n"
            
            for i, exp in enumerate(explanations[:3], 1):
                if exp['impact_direction'] == "increases":
                    summary += f"{i}. {exp['feature']} (value: {exp['value']:.2f}) significantly increases fraud risk\n"
            
            summary += "\nRecommendation: Manual review required before claim approval."
            
        else:
            summary = "This claim appears GENUINE based on the following factors:\n\n"
            
            for i, exp in enumerate(explanations[:3], 1):
                summary += f"{i}. {exp['feature']} (value: {exp['value']:.2f}) supports legitimacy\n"
            
            summary += "\nRecommendation: Claim can be processed normally."
        
        return summary
    
    def get_feature_importance(self):
        """
        Get global feature importance from XGBoost model
        
        Returns:
            dict: Feature importance scores
        """
        importance_scores = self.xgb_model.feature_importances_
        
        feature_importance = []
        for name, score in zip(self.feature_names, importance_scores):
            readable_name = self._get_readable_feature_name(name)
            feature_importance.append({
                'feature': readable_name,
                'importance': float(score)
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return feature_importance
