#!/usr/bin/env python3
"""
Test script for batch processing functionality
"""

import requests
import json
import time

# Test data - multiple claims
test_claims = [
    {
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
    },
    {
        "age": 65,
        "gender": 1,
        "claim_amount": 50000.00,
        "hospital_stay_days": 15,
        "previous_claims": 8,
        "treatment_type": 2,
        "provider_type": 1,
        "diagnosis_code": 780,
        "procedure_code": 8500,
        "chronic_condition": 1,
        "insurance_type": 2,
        "policy_age_days": 2000,
        "beneficiaries": 1
    },
    {
        "age": 25,
        "gender": 1,
        "claim_amount": 1500.00,
        "hospital_stay_days": 1,
        "previous_claims": 0,
        "treatment_type": 0,
        "provider_type": 2,
        "diagnosis_code": 150,
        "procedure_code": 1200,
        "chronic_condition": 0,
        "insurance_type": 0,
        "policy_age_days": 300,
        "beneficiaries": 4
    }
]

def test_batch_processing():
    """Test the batch processing endpoint"""
    url = "http://localhost:8000/batch-predict"

    print("Testing batch processing with", len(test_claims), "claims...")
    print("=" * 60)

    try:
        # Make request
        start_time = time.time()
        response = requests.post(url, json=test_claims, timeout=30)
        end_time = time.time()

        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\nBatch processing successful!")
            print(f"Total claims: {result['total_claims']}")
            print(f"Processed claims: {result['processed_claims']}")
            print(f"Failed claims: {result['failed_claims']}")
            print(f"Fraud predictions: {result['summary']['fraud_predictions']}")
            print(".1f")

            print("\nDetailed results:")
            for i, claim_result in enumerate(result['results']):
                if claim_result['success']:
                    print(f"Claim {i+1}: {claim_result['final_prediction']} "
                          f"(Risk: {claim_result['hybrid_risk_score']:.3f}, "
                          f"Confidence: {claim_result['confidence']:.1%})")
                else:
                    print(f"Claim {i+1}: FAILED - {claim_result['error']}")

        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to the API. Make sure the backend is running.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_batch_processing()