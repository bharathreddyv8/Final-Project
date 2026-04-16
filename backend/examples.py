"""
Example API Requests and Responses
For testing the Medical Insurance Fraud Detection API
"""

# =============================================================================
# EXAMPLE 1: Legitimate/Genuine Claim
# =============================================================================

REQUEST_LEGITIMATE = {
    "age": 35,
    "gender": 0,
    "claim_amount": 5000.00,
    "hospital_stay_days": 3,
    "previous_claims": 1,
    "treatment_type": 0,
    "provider_type": 0,
    "diagnosis_code": 250,
    "procedure_code": 2500,
    "chronic_condition": 0,
    "insurance_type": 1,
    "policy_age_days": 1000,
    "beneficiaries": 2
}

# Expected Response: GENUINE
RESPONSE_LEGITIMATE = {
    "success": True,
    "timestamp": "2026-02-02T10:30:00.123456",
    "xgboost_probability": 0.1234,
    "isolation_forest_score": 0.0987,
    "hybrid_risk_score": 0.1160,
    "final_prediction": "GENUINE",
    "confidence": 0.8840,
    "risk_level": "LOW",
    "explanation": [
        {
            "feature": "Claim Amount",
            "value": 5000.0,
            "impact": -0.0234,
            "impact_direction": "decreases",
            "absolute_impact": 0.0234
        },
        {
            "feature": "Number of Previous Claims",
            "value": 1.0,
            "impact": -0.0189,
            "impact_direction": "decreases",
            "absolute_impact": 0.0189
        }
    ],
    "summary": "This claim appears GENUINE based on the following factors...",
    "model_info": {
        "xgboost_weight": 0.7,
        "isolation_forest_weight": 0.3,
        "decision_threshold": 0.5,
        "hybrid_approach": "Weighted combination of supervised and unsupervised models"
    }
}


# =============================================================================
# EXAMPLE 2: Fraudulent Claim - High Amount, Short Stay
# =============================================================================

REQUEST_FRAUD_HIGH_AMOUNT = {
    "age": 28,
    "gender": 1,
    "claim_amount": 48000.00,
    "hospital_stay_days": 1,
    "previous_claims": 15,
    "treatment_type": 2,
    "provider_type": 2,
    "diagnosis_code": 900,
    "procedure_code": 9500,
    "chronic_condition": 1,
    "insurance_type": 0,
    "policy_age_days": 15,
    "beneficiaries": 8
}

# Expected Response: FRAUD
RESPONSE_FRAUD_HIGH_AMOUNT = {
    "success": True,
    "timestamp": "2026-02-02T10:35:00.123456",
    "xgboost_probability": 0.8567,
    "isolation_forest_score": 0.7890,
    "hybrid_risk_score": 0.8364,
    "final_prediction": "FRAUD",
    "confidence": 0.8364,
    "risk_level": "CRITICAL",
    "explanation": [
        {
            "feature": "Claim Amount",
            "value": 48000.0,
            "impact": 0.2345,
            "impact_direction": "increases",
            "absolute_impact": 0.2345
        },
        {
            "feature": "Number of Previous Claims",
            "value": 15.0,
            "impact": 0.1876,
            "impact_direction": "increases",
            "absolute_impact": 0.1876
        },
        {
            "feature": "Policy Age (Days)",
            "value": 15.0,
            "impact": 0.1654,
            "impact_direction": "increases",
            "absolute_impact": 0.1654
        }
    ],
    "summary": "This claim is flagged as FRAUD due to the following suspicious factors...",
    "model_info": {
        "xgboost_weight": 0.7,
        "isolation_forest_weight": 0.3,
        "decision_threshold": 0.5,
        "hybrid_approach": "Weighted combination of supervised and unsupervised models"
    }
}


# =============================================================================
# EXAMPLE 3: Medium Risk Claim
# =============================================================================

REQUEST_MEDIUM_RISK = {
    "age": 52,
    "gender": 1,
    "claim_amount": 22000.00,
    "hospital_stay_days": 4,
    "previous_claims": 5,
    "treatment_type": 1,
    "provider_type": 0,
    "diagnosis_code": 550,
    "procedure_code": 5500,
    "chronic_condition": 1,
    "insurance_type": 1,
    "policy_age_days": 500,
    "beneficiaries": 4
}

# Expected Response: Could be either, depends on model
RESPONSE_MEDIUM_RISK = {
    "success": True,
    "timestamp": "2026-02-02T10:40:00.123456",
    "xgboost_probability": 0.4567,
    "isolation_forest_score": 0.3890,
    "hybrid_risk_score": 0.4364,
    "final_prediction": "GENUINE",
    "confidence": 0.5636,
    "risk_level": "MEDIUM",
    "explanation": [
        {
            "feature": "Claim Amount",
            "value": 22000.0,
            "impact": 0.0876,
            "impact_direction": "increases",
            "absolute_impact": 0.0876
        },
        {
            "feature": "Chronic Condition",
            "value": 1.0,
            "impact": 0.0654,
            "impact_direction": "increases",
            "absolute_impact": 0.0654
        }
    ],
    "summary": "This claim appears GENUINE based on the following factors...",
    "model_info": {
        "xgboost_weight": 0.7,
        "isolation_forest_weight": 0.3,
        "decision_threshold": 0.5,
        "hybrid_approach": "Weighted combination of supervised and unsupervised models"
    }
}


# =============================================================================
# EXAMPLE 4: Emergency Treatment - Legitimate
# =============================================================================

REQUEST_EMERGENCY_LEGITIMATE = {
    "age": 67,
    "gender": 0,
    "claim_amount": 15000.00,
    "hospital_stay_days": 7,
    "previous_claims": 2,
    "treatment_type": 2,
    "provider_type": 0,
    "diagnosis_code": 410,
    "procedure_code": 4100,
    "chronic_condition": 1,
    "insurance_type": 2,
    "policy_age_days": 2500,
    "beneficiaries": 1
}


# =============================================================================
# EXAMPLE 5: Fraud network update
# =============================================================================

if __name__ == "__main__":
    # Demonstrate updating the fraud network separately from the API
    from model.fraud_network import update_fraud_network

    example_claim = REQUEST_FRAUD_HIGH_AMOUNT.copy()
    example_claim["patient_id"] = "PATIENT123"
    graph, suspicious, clusters = update_fraud_network(example_claim, fraud_prediction=1)
    print("Suspicious nodes after one update:", suspicious)
    print("Detected communities:", clusters)


# =============================================================================
# EXAMPLE 5: Fraudulent - New Policy High Claim
# =============================================================================

REQUEST_FRAUD_NEW_POLICY = {
    "age": 32,
    "gender": 1,
    "claim_amount": 38000.00,
    "hospital_stay_days": 2,
    "previous_claims": 8,
    "treatment_type": 1,
    "provider_type": 1,
    "diagnosis_code": 780,
    "procedure_code": 8900,
    "chronic_condition": 0,
    "insurance_type": 0,
    "policy_age_days": 10,
    "beneficiaries": 6
}


# =============================================================================
# CURL EXAMPLES FOR TESTING
# =============================================================================

CURL_LEGITIMATE = """
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "age": 35,
    "gender": 0,
    "claim_amount": 5000.00,
    "hospital_stay_days": 3,
    "previous_claims": 1,
    "treatment_type": 0,
    "provider_type": 0,
    "diagnosis_code": 250,
    "procedure_code": 2500,
    "chronic_condition": 0,
    "insurance_type": 1,
    "policy_age_days": 1000,
    "beneficiaries": 2
  }'
"""

CURL_FRAUD = """
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "age": 28,
    "gender": 1,
    "claim_amount": 48000.00,
    "hospital_stay_days": 1,
    "previous_claims": 15,
    "treatment_type": 2,
    "provider_type": 2,
    "diagnosis_code": 900,
    "procedure_code": 9500,
    "chronic_condition": 1,
    "insurance_type": 0,
    "policy_age_days": 15,
    "beneficiaries": 8
  }'
"""

# =============================================================================
# PYTHON REQUESTS EXAMPLE
# =============================================================================

PYTHON_EXAMPLE = """
import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Claim data
claim_data = {
    "age": 45,
    "gender": 1,
    "claim_amount": 25000.00,
    "hospital_stay_days": 5,
    "previous_claims": 3,
    "treatment_type": 1,
    "provider_type": 0,
    "diagnosis_code": 450,
    "procedure_code": 5678,
    "chronic_condition": 0,
    "insurance_type": 1,
    "policy_age_days": 730,
    "beneficiaries": 3
}

# Make prediction request
response = requests.post(url, json=claim_data)

# Get result
result = response.json()

# Display prediction
print(f"Prediction: {result['final_prediction']}")
print(f"Risk Score: {result['hybrid_risk_score']:.2%}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"\\nTop Contributing Features:")
for exp in result['explanation'][:3]:
    print(f"  - {exp['feature']}: {exp['impact']:.4f}")
"""


# =============================================================================
# JAVASCRIPT FETCH EXAMPLE
# =============================================================================

JAVASCRIPT_EXAMPLE = """
// Fraud detection API call
async function predictFraud(claimData) {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(claimData)
    });
    
    const result = await response.json();
    
    console.log('Prediction:', result.final_prediction);
    console.log('Risk Score:', result.hybrid_risk_score);
    console.log('XGBoost:', result.xgboost_probability);
    console.log('Isolation Forest:', result.isolation_forest_score);
    
    return result;
}

// Example usage
const claim = {
    age: 45,
    gender: 1,
    claim_amount: 25000.00,
    hospital_stay_days: 5,
    previous_claims: 3,
    treatment_type: 1,
    provider_type: 0,
    diagnosis_code: 450,
    procedure_code: 5678,
    chronic_condition: 0,
    insurance_type: 1,
    policy_age_days: 730,
    beneficiaries: 3
};

predictFraud(claim);
"""


# =============================================================================
# VALIDATION ERROR EXAMPLE
# =============================================================================

INVALID_REQUEST = {
    "age": 150,  # Invalid: too old
    "gender": 3,  # Invalid: should be 0 or 1
    "claim_amount": -1000,  # Invalid: negative
    "hospital_stay_days": 400  # Invalid: too many days
}

VALIDATION_ERROR_RESPONSE = {
    "detail": [
        {
            "loc": ["body", "age"],
            "msg": "ensure this value is less than or equal to 120",
            "type": "value_error.number.not_le"
        },
        {
            "loc": ["body", "claim_amount"],
            "msg": "ensure this value is greater than 0",
            "type": "value_error.number.not_gt"
        }
    ]
}
