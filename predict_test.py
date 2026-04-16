import requests, json
url='http://localhost:8000/predict'
claim={'age':45,'gender':1,'claim_amount':25000.0,'hospital_stay_days':5,'previous_claims':3,'treatment_type':1,'provider_type':0,'diagnosis_code':450,'procedure_code':5678,'chronic_condition':0,'insurance_type':1,'policy_age_days':730,'beneficiaries':3}
try:
    r=requests.post(url,json=claim,timeout=30)
    print('status', r.status_code)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print('error', e)
