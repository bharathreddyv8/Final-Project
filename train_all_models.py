import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load dataset
df = pd.read_csv('backend/data/insurance_claims_dataset.csv')

# Prepare features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}

# 1. Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
}

# 2. Decision Tree
print("Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred = dt.predict(X_test_scaled)
y_pred_proba = dt.predict_proba(X_test_scaled)[:, 1]
results['Decision Tree'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
}

# 3. Naive Bayes
print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred = nb.predict(X_test_scaled)
y_pred_proba = nb.predict_proba(X_test_scaled)[:, 1]
results['Naive Bayes'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
}

# 4. Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
results['Random Forest'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
}

# 5. XGBoost
print("Training XGBoost...")
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
xgb.fit(X_train_scaled, y_train)
y_pred = xgb.predict(X_test_scaled)
y_pred_proba = xgb.predict_proba(X_test_scaled)[:, 1]
results['XGBoost'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
}

# 6. Isolation Forest
print("Training Isolation Forest...")
iso = IsolationForest(random_state=42)
iso.fit(X_train_scaled)
iso_scores_test = iso.score_samples(X_test_scaled)
iso_pred = (iso_scores_test < np.percentile(iso_scores_test, 50)).astype(int)
iso_pred_proba = 1 - (iso_scores_test - iso_scores_test.min()) / (iso_scores_test.max() - iso_scores_test.min() + 1e-10)
results['Isolation Forest'] = {
    'Accuracy': accuracy_score(y_test, iso_pred),
    'Precision': precision_score(y_test, iso_pred, zero_division=0),
    'Recall': recall_score(y_test, iso_pred, zero_division=0),
    'F1 Score': f1_score(y_test, iso_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, iso_pred_proba)
}

# 7. Hybrid Model
print("Creating Hybrid Model...")
xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
iso_hybrid_proba = 1 - (iso_scores_test - iso_scores_test.min()) / (iso_scores_test.max() - iso_scores_test.min() + 1e-10)
hybrid_score = 0.7 * xgb_proba + 0.3 * iso_hybrid_proba
hybrid_pred = (hybrid_score >= 0.5).astype(int)
results['Hybrid Model'] = {
    'Accuracy': accuracy_score(y_test, hybrid_pred),
    'Precision': precision_score(y_test, hybrid_pred),
    'Recall': recall_score(y_test, hybrid_pred),
    'F1 Score': f1_score(y_test, hybrid_pred),
    'ROC-AUC': roc_auc_score(y_test, hybrid_score)
}

# Print results
print("\n" + "="*80)
print("REAL MODEL EVALUATION RESULTS")
print("="*80)
for model, metrics in results.items():
    print(f"\n{model}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")

# Save to JSON
import json
metrics_json = {}
for model, metrics in results.items():
    metrics_json[model] = {k: float(v) for k, v in metrics.items()}

with open('real_model_metrics.json', 'w') as f:
    json.dump(metrics_json, f, indent=2)

print("\n✅ Results saved to real_model_metrics.json")
