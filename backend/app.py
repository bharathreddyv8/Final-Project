"""
FastAPI Backend for Medical Insurance Fraud Detection
Author: Bharath Kumar
Academic Project - Real-Time Fraud Detection with Hybrid ML Models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from datetime import datetime
import json
import os
import requests

# File to persist recent predictions so history survives server restarts
RECENT_PREDICTIONS_FILE = os.path.join(os.path.dirname(__file__), 'recent_predictions.json')

# Import prediction module
try:
    # Works when running from backend/ directory
    from model.predict import get_predictor
except ModuleNotFoundError:
    # Works when running from project root as package path
    from backend.model.predict import get_predictor

# Initialize FastAPI app
app = FastAPI(
    title="Medical Insurance Fraud Detection API",
    description="Real-time fraud detection using XGBoost + Isolation Forest",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance (loaded once)
predictor = None

# Store recent predictions for admin dashboard
recent_predictions = []
MAX_RECENT_PREDICTIONS = 100


def _load_recent_predictions_from_disk():
    global recent_predictions
    try:
        if os.path.exists(RECENT_PREDICTIONS_FILE):
            with open(RECENT_PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    recent_predictions = data
                else:
                    recent_predictions = []
            # Trim to max size if file has grown
            if len(recent_predictions) > MAX_RECENT_PREDICTIONS:
                recent_predictions = recent_predictions[-MAX_RECENT_PREDICTIONS:]
            print(f"Loaded {len(recent_predictions)} recent predictions from disk")
        else:
            recent_predictions = []
    except Exception as e:
        print(f"Error loading recent predictions: {e}")
        recent_predictions = []


def _save_recent_predictions_to_disk():
    try:
        with open(RECENT_PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(recent_predictions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving recent predictions: {e}")


def _build_provider_id(claim_data: Dict[str, Any]) -> str:
    """
    Build a deterministic provider identifier from claim attributes.
    """
    provider_prefix = {
        0: "HOSP",
        1: "CLIN",
        2: "INDV"
    }.get(claim_data.get("provider_type", 0), "PROV")
    provider_hash = (
        claim_data.get("procedure_code", 0) * 31
        + claim_data.get("diagnosis_code", 0) * 17
        + claim_data.get("policy_age_days", 0)
    ) % 10000
    return f"PRV-{provider_prefix}-{provider_hash:04d}"


def _build_patient_id(claim_data: Dict[str, Any]) -> str:
    """
    Build a deterministic patient identifier from claim attributes.
    """
    patient_hash = (
        claim_data.get("age", 0) * 29
        + claim_data.get("beneficiaries", 0) * 13
        + claim_data.get("previous_claims", 0) * 7
        + claim_data.get("chronic_condition", 0) * 97
    ) % 10000
    gender_prefix = "M" if claim_data.get("gender", 0) == 1 else "F"
    return f"PAT-{gender_prefix}-{patient_hash:04d}"


def send_email(to_email: str, subject: str, message: str) -> (bool, Optional[str]):
    """
    Send email via Gmail SMTP. Uses environment variables `SMTP_EMAIL` and `SMTP_PASSWORD`.
    Returns (True, None) on success or (False, error_message) on failure.
    """
    try:
        import smtplib
        import ssl

        sender_email = os.getenv('SMTP_EMAIL')
        sender_password = os.getenv('SMTP_PASSWORD')
        if not sender_email or not sender_password:
            return False, 'SMTP credentials not configured (SMTP_EMAIL/SMTP_PASSWORD)'

        smtp_server = 'smtp.gmail.com'
        port = 587
        context = ssl.create_default_context()

        full_message = f"From: {sender_email}\r\nTo: {to_email}\r\nSubject: {subject}\r\n\r\n{message}"

        with smtplib.SMTP(smtp_server, port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, full_message.encode('utf-8'))

        return True, None
    except Exception as e:
        print(f"Error sending email: {e}")
        return False, str(e)


def send_alerts(claim_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send alerts for high-risk (FRAUD) events via configured channels.

    Configurable via environment variables:
      - ALERT_EMAILS: comma-separated admin email addresses
      - SLACK_WEBHOOK_URL: slack incoming webhook URL
      - TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, ALERT_SMS_NUMBERS (comma-separated)

    Returns a dict with status per channel and overall success flag.
    """
    results = {
        'alert_sent': False,
        'channels': [],
        'errors': []
    }

    subject = f"FRAUD Alert - Claim"
    body_lines = [
        "FRAUD alert detected by Fraud Detection system",
        "",
        f"Final prediction: {result.get('final_prediction')}",
        f"Hybrid risk score: {result.get('hybrid_risk_score')}",
        "",
        "Claim details:",
    ]
    for k, v in claim_data.items():
        body_lines.append(f"- {k}: {v}")

    message = "\n".join(body_lines)

    # 1) Email alerts to admin emails
    alert_emails = os.getenv('ALERT_EMAILS')
    if alert_emails:
        for addr in [e.strip() for e in alert_emails.split(',') if e.strip()]:
            sent, err = send_email(addr, subject, message)
            if sent:
                results['channels'].append('email')
                results['alert_sent'] = True
            else:
                results['errors'].append(f"email:{addr}:{err}")

    # 2) Slack webhook
    slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
    if slack_webhook:
        try:
            resp = requests.post(slack_webhook, json={"text": message}, timeout=10)
            if resp.status_code == 200:
                results['channels'].append('slack')
                results['alert_sent'] = True
            else:
                results['errors'].append(f"slack:{resp.status_code}:{resp.text}")
        except Exception as e:
            results['errors'].append(f"slack:{str(e)}")

    # 3) Twilio SMS
    tw_sid = os.getenv('TWILIO_ACCOUNT_SID')
    tw_token = os.getenv('TWILIO_AUTH_TOKEN')
    tw_from = os.getenv('TWILIO_FROM_NUMBER')
    alert_sms = os.getenv('ALERT_SMS_NUMBERS')
    if tw_sid and tw_token and tw_from and alert_sms:
        for to_number in [n.strip() for n in alert_sms.split(',') if n.strip()]:
            try:
                url = f"https://api.twilio.com/2010-04-01/Accounts/{tw_sid}/Messages.json"
                payload = {
                    'From': tw_from,
                    'To': to_number,
                    'Body': message[:1600]
                }
                resp = requests.post(url, data=payload, auth=(tw_sid, tw_token), timeout=10)
                if resp.status_code in (200, 201):
                    results['channels'].append('sms')
                    results['alert_sent'] = True
                else:
                    results['errors'].append(f"twilio:{to_number}:{resp.status_code}:{resp.text}")
            except Exception as e:
                results['errors'].append(f"twilio:{to_number}:{str(e)}")

    return results


# Pydantic models for request/response validation
class ClaimInput(BaseModel):
    """
    Medical insurance claim input schema
    """
    age: int = Field(..., ge=18, le=120, description="Patient age (18-120)")
    gender: int = Field(..., ge=0, le=1, description="Gender (0: Female, 1: Male)")
    claim_amount: float = Field(..., gt=0, description="Claim amount in dollars")
    hospital_stay_days: int = Field(..., ge=0, le=365, description="Hospital stay duration in days")
    previous_claims: int = Field(..., ge=0, description="Number of previous claims")
    treatment_type: int = Field(..., ge=0, le=2, description="Treatment type (0: Outpatient, 1: Inpatient, 2: Emergency)")
    provider_type: int = Field(..., ge=0, le=2, description="Provider type (0: Hospital, 1: Clinic, 2: Individual)")
    diagnosis_code: int = Field(..., ge=100, le=999, description="Diagnosis code (100-999)")
    procedure_code: int = Field(..., ge=1000, le=9999, description="Procedure code (1000-9999)")
    chronic_condition: int = Field(..., ge=0, le=1, description="Chronic condition (0: No, 1: Yes)")
    insurance_type: int = Field(..., ge=0, le=2, description="Insurance type (0: Basic, 1: Premium, 2: Gold)")
    policy_age_days: int = Field(..., ge=1, description="Days since policy start")
    beneficiaries: int = Field(..., ge=1, le=20, description="Number of beneficiaries")
    patient_email: Optional[str] = Field(None, description="Optional patient email for notification")
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class FeatureExplanation(BaseModel):
    """
    Feature explanation schema
    """
    feature: str
    value: float
    impact: float
    impact_direction: str
    absolute_impact: float


class PredictionResponse(BaseModel):
    """
    Prediction response schema
    """
    success: bool
    timestamp: str
    xgboost_probability: float
    isolation_forest_score: float
    hybrid_risk_score: float
    final_prediction: str
    confidence: float
    risk_level: str
    explanation: List[FeatureExplanation]
    summary: str
    model_info: Dict[str, Any]
    # Email send status (optional)
    email_sent: Optional[bool] = False
    email_to: Optional[str] = None
    email_error: Optional[str] = None
    # Alerting status for FRAUD notifications
    alert_sent: Optional[bool] = False
    alert_channels: Optional[List[str]] = None
    alert_errors: Optional[List[str]] = None


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response schema
    """
    success: bool
    timestamp: str
    total_claims: int
    processed_claims: int
    failed_claims: int
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]


class HealthResponse(BaseModel):
    """
    Health check response
    """
    status: str
    message: str
    models_loaded: bool
    timestamp: str


class RecentPrediction(BaseModel):
    """
    Recent prediction record
    """
    claim_id: str
    timestamp: str
    prediction: str
    risk_score: float
    claim_amount: float


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """
    Load models on startup
    """
    global predictor
    # Load persisted recent predictions so admin dashboard shows history
    _load_recent_predictions_from_disk()
    try:
        print("Loading ML models...")
        predictor = get_predictor()
        print("✓ Models loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")
        print("Please run 'python backend/model/train.py' first to train models.")


@app.get("/", response_model=HealthResponse)
async def root():
    """
    Root endpoint - health check
    """
    return {
        "status": "online",
        "message": "Medical Insurance Fraud Detection API",
        "models_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Detailed health check endpoint
    """
    models_loaded = predictor is not None
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "message": "All systems operational" if models_loaded else "Models not loaded",
        "models_loaded": models_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(claim: ClaimInput):
    """
    Main prediction endpoint
    
    Accepts a medical insurance claim and returns fraud prediction with explanation
    """
    global recent_predictions
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train models first."
        )
    
    try:
        # Convert claim to dictionary
        claim_data = claim.dict()
        
        # Validate input
        is_valid, errors = predictor.validate_input(claim_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail={"errors": errors})
        
        # Make prediction
        result = predictor.predict(claim_data)
        
        # Generate claim ID
        claim_id = f"CLM{len(recent_predictions):06d}"
        timestamp = datetime.now().isoformat()
        
        # Store in recent predictions
        recent_predictions.append({
            "claim_id": claim_id,
            "timestamp": timestamp,
            "prediction": result['final_prediction'],
            "risk_score": result['hybrid_risk_score'],
            "claim_amount": claim_data['claim_amount'],
            "provider_id": _build_provider_id(claim_data),
            "patient_id": _build_patient_id(claim_data)
        })
        
        # Keep only recent N predictions
        if len(recent_predictions) > MAX_RECENT_PREDICTIONS:
            recent_predictions.pop(0)
        # Persist updated recent predictions
        _save_recent_predictions_to_disk()
        
        # Prepare response
        response = {
            "success": True,
            "timestamp": timestamp,
            "xgboost_probability": result['xgboost_probability'],
            "isolation_forest_score": result['isolation_forest_score'],
            "hybrid_risk_score": result['hybrid_risk_score'],
            "final_prediction": result['final_prediction'],
            "confidence": result['confidence'],
            "risk_level": result['risk_level'],
            "explanation": result['explanation'],
            "summary": result['summary'],
            "model_info": {
                "xgboost_weight": 0.7,
                "isolation_forest_weight": 0.3,
                "decision_threshold": 0.5,
                "hybrid_approach": "Weighted combination of supervised and unsupervised models"
            }
        }

        # If patient provided an email and claim is approved, send an approval notification
        email_sent = False
        email_to = None
        email_error = None
        alert_info = None
        try:
            if claim_data.get('patient_email') and result.get('final_prediction') == 'GENUINE':
                email_to = str(claim_data.get('patient_email'))
                subject = "Claim Approved"
                message = (
                    f"Dear Patient,\n\n"
                    f"Your insurance claim has been approved.\n\n"
                    f"Claim ID: {claim_id}\n"
                    f"Risk Score: {result.get('hybrid_risk_score', 0) * 100:.1f}%\n\n"
                    f"If you have any questions, please contact support.\n\n"
                    f"Regards,\nInsurance Team"
                )
                sent, err = send_email(email_to, subject, message)
                email_sent = bool(sent)
                email_error = err
                if sent:
                    print(f"Approval email sent to {email_to}")
                else:
                    print(f"Failed to send approval email to {email_to}: {err}")
        except Exception as e:
            print(f"Unexpected error during email send: {e}")
        # If FRAUD detected, send alerts to configured channels
        try:
            if result.get('final_prediction') == 'FRAUD':
                alert_info = send_alerts(claim_data, result)
                if alert_info.get('alert_sent'):
                    print(f"Alerts sent for claim {claim_id}: {alert_info.get('channels')}")
                else:
                    print(f"Alert errors: {alert_info.get('errors')}")
        except Exception as e:
            print(f"Unexpected error sending alerts: {e}")

        # Include email status in response (frontend can display notification)
        response['email_sent'] = email_sent
        response['email_to'] = email_to
        response['email_error'] = email_error
        # Include alert info
        response['alert_sent'] = bool(alert_info and alert_info.get('alert_sent'))
        response['alert_channels'] = alert_info.get('channels') if alert_info else []
        response['alert_errors'] = alert_info.get('errors') if alert_info else []
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict_fraud(claims: List[ClaimInput]):
    """
    Batch prediction endpoint
    
    Accepts multiple medical insurance claims and returns fraud predictions with explanations
    """
    global recent_predictions
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train models first."
        )
    
    if not claims:
        raise HTTPException(
            status_code=400,
            detail="No claims provided"
        )
    
    if len(claims) > 1000:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 claims allowed per batch"
        )
    
    try:
        # Convert claims to dictionaries
        claims_data = [claim.dict() for claim in claims]
        
        # Make batch predictions
        batch_results = predictor.predict_batch(claims_data)
        
        # Generate claim IDs and timestamps
        timestamp = datetime.now().isoformat()
        base_claim_id = len(recent_predictions)
        
        # Process results and add to recent predictions
        processed_results = []
        fraud_count = 0
        genuine_count = 0
        total_risk_score = 0
        total_claim_amount = 0
        
        for i, result in enumerate(batch_results):
            if result['success']:
                # Generate claim ID
                claim_id = f"CLM{base_claim_id + i:06d}"
                
                # Store in recent predictions
                recent_predictions.append({
                    "claim_id": claim_id,
                    "timestamp": timestamp,
                    "prediction": result['final_prediction'],
                    "risk_score": result['hybrid_risk_score'],
                    "claim_amount": claims_data[i]['claim_amount'],
                    "provider_id": _build_provider_id(claims_data[i]),
                    "patient_id": _build_patient_id(claims_data[i])
                })
                
                # Update statistics
                if result['final_prediction'] == 'FRAUD':
                    fraud_count += 1
                else:
                    genuine_count += 1
                
                total_risk_score += result['hybrid_risk_score']
                total_claim_amount += claims_data[i]['claim_amount']
                
                # Add claim_id to result
                result['claim_id'] = claim_id
                result['timestamp'] = timestamp
                # If FRAUD, send alerts and attach info
                if result.get('final_prediction') == 'FRAUD':
                    try:
                        alert_info = send_alerts(claims_data[i], result)
                        result['alert_sent'] = bool(alert_info.get('alert_sent'))
                        result['alert_channels'] = alert_info.get('channels')
                        result['alert_errors'] = alert_info.get('errors')
                    except Exception as e:
                        result['alert_sent'] = False
                        result['alert_channels'] = []
                        result['alert_errors'] = [str(e)]
            
            processed_results.append(result)
        
        # Keep only recent N predictions
        if len(recent_predictions) > MAX_RECENT_PREDICTIONS:
            recent_predictions = recent_predictions[-MAX_RECENT_PREDICTIONS:]
        # Persist updated recent predictions after batch processing
        _save_recent_predictions_to_disk()
        
        # Calculate summary statistics
        processed_count = len([r for r in batch_results if r['success']])
        failed_count = len(claims) - processed_count
        
        summary = {
            "total_claims": len(claims),
            "processed_claims": processed_count,
            "failed_claims": failed_count,
            "fraud_predictions": fraud_count,
            "genuine_predictions": genuine_count,
            "fraud_rate": (fraud_count / processed_count * 100) if processed_count > 0 else 0,
            "avg_risk_score": (total_risk_score / processed_count) if processed_count > 0 else 0,
            "avg_claim_amount": (total_claim_amount / processed_count) if processed_count > 0 else 0
        }
        
        # Prepare response
        response = {
            "success": True,
            "timestamp": timestamp,
            "total_claims": len(claims),
            "processed_claims": processed_count,
            "failed_claims": failed_count,
            "results": processed_results,
            "summary": summary
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.get("/recent-predictions")
async def get_recent_predictions(limit: int = 20):
    """
    Get recent predictions for admin dashboard
    """
    return {
        "success": True,
        "count": len(recent_predictions),
        "predictions": recent_predictions[-limit:][::-1]  # Most recent first
    }


@app.delete("/recent-predictions")
async def clear_recent_predictions():
    """
    Clear stored recent predictions (in-memory and on disk).
    """
    global recent_predictions
    try:
        recent_predictions = []
        _save_recent_predictions_to_disk()
        return {
            "success": True,
            "message": "Recent predictions cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing recent predictions: {e}")


@app.get("/statistics")
async def get_statistics():
    """
    Get prediction statistics
    """
    if not recent_predictions:
        return {
            "success": True,
            "message": "No predictions yet",
            "total_predictions": 0
        }
    
    fraud_count = sum(1 for p in recent_predictions if p['prediction'] == 'FRAUD')
    genuine_count = len(recent_predictions) - fraud_count
    
    avg_risk_score = sum(p['risk_score'] for p in recent_predictions) / len(recent_predictions)
    avg_claim_amount = sum(p['claim_amount'] for p in recent_predictions) / len(recent_predictions)
    
    return {
        "success": True,
        "total_predictions": len(recent_predictions),
        "fraud_count": fraud_count,
        "genuine_count": genuine_count,
        "fraud_rate": fraud_count / len(recent_predictions) * 100,
        "avg_risk_score": avg_risk_score,
        "avg_claim_amount": avg_claim_amount,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/fraud-risk-leaderboard")
async def get_fraud_risk_leaderboard(entity_type: str = "provider", limit: int = 10):
    """
    Get top risky providers or patients ranked by fraud risk.
    """
    if entity_type not in {"provider", "patient"}:
        raise HTTPException(
            status_code=400,
            detail="entity_type must be either 'provider' or 'patient'"
        )
    if limit < 1 or limit > 50:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 50"
        )

    if not recent_predictions:
        return {
            "success": True,
            "entity_type": entity_type,
            "leaderboard": [],
            "count": 0,
            "timestamp": datetime.now().isoformat()
        }

    key = "provider_id" if entity_type == "provider" else "patient_id"
    prefix = "PRV-UNKNOWN" if entity_type == "provider" else "PAT-UNKNOWN"

    aggregated: Dict[str, Dict[str, Any]] = {}

    for prediction in recent_predictions:
        entity_id = prediction.get(key) or prefix
        risk = float(prediction.get("risk_score", 0))
        is_fraud = prediction.get("prediction") == "FRAUD"

        if entity_id not in aggregated:
            aggregated[entity_id] = {
                "entity_id": entity_id,
                "total_claims": 0,
                "fraud_claims": 0,
                "risk_sum": 0.0
            }

        aggregated[entity_id]["total_claims"] += 1
        aggregated[entity_id]["fraud_claims"] += 1 if is_fraud else 0
        aggregated[entity_id]["risk_sum"] += risk

    ranked_items = []
    for entity_id, stats in aggregated.items():
        total_claims = stats["total_claims"]
        avg_risk = stats["risk_sum"] / total_claims if total_claims else 0.0
        fraud_rate = stats["fraud_claims"] / total_claims if total_claims else 0.0
        ranking_score = (0.7 * avg_risk) + (0.3 * fraud_rate)

        ranked_items.append({
            "entity_id": entity_id,
            "risk_score": round(ranking_score, 4),
            "average_risk_score": round(avg_risk, 4),
            "fraud_rate": round(fraud_rate, 4),
            "total_claims": total_claims
        })

    ranked_items.sort(
        key=lambda item: (
            item["risk_score"],
            item["average_risk_score"],
            item["total_claims"]
        ),
        reverse=True
    )

    leaderboard = []
    for idx, item in enumerate(ranked_items[:limit], start=1):
        leaderboard.append({
            "rank": idx,
            "name": item["entity_id"],
            "risk_score": item["risk_score"],
            "average_risk_score": item["average_risk_score"],
            "fraud_rate": item["fraud_rate"],
            "total_claims": item["total_claims"]
        })

    return {
        "success": True,
        "entity_type": entity_type,
        "leaderboard": leaderboard,
        "count": len(leaderboard),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info")
async def get_model_info():
    """
    Get information about the models
    """
    return {
        "success": True,
        "models": {
            "xgboost": {
                "type": "Supervised Classification",
                "purpose": "Fraud probability prediction",
                "weight": 0.7
            },
            "isolation_forest": {
                "type": "Unsupervised Anomaly Detection",
                "purpose": "Anomaly score calculation",
                "weight": 0.3
            }
        },
        "hybrid_logic": {
            "description": "Weighted combination of XGBoost and Isolation Forest",
            "formula": "hybrid_score = 0.7 * xgboost_prob + 0.3 * anomaly_score",
            "threshold": 0.5
        },
        "explainability": {
            "method": "SHAP (SHapley Additive exPlanations)",
            "purpose": "Feature contribution analysis"
        }
    }


@app.get("/evaluation-metrics")
async def get_evaluation_metrics():
    """
    Get model evaluation metrics
    """
    try:
        metrics_path = os.path.join(os.path.dirname(__file__), 'models', 'evaluation_metrics.json')
        
        if not os.path.exists(metrics_path):
            return {
                "success": False,
                "message": "Evaluation metrics not found. Please train the model first."
            }
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return {
            "success": True,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading metrics: {str(e)}"
        )


# Run the application
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MEDICAL INSURANCE FRAUD DETECTION API")
    print("Student: Bharath Kumar")
    print("="*70)
    print("\nStarting FastAPI server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Frontend URL: Open frontend/index.html in browser")
    print("="*70 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
