import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle

print("Loading data...")
df = pd.read_excel('insurance_claims.xlsx')

# Prepare the data
print("Preparing data...")

# Separate features and target
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported'].map({'Y': 1, 'N': 0})  # Convert Y/N to 1/0

# Handle categorical columns - convert to numeric
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Handle date columns - extract useful features
if 'policy_bind_date' in X.columns:
    X['policy_bind_year'] = pd.to_datetime(X['policy_bind_date']).dt.year
    X['policy_bind_month'] = pd.to_datetime(X['policy_bind_date']).dt.month
    X = X.drop('policy_bind_date', axis=1)

if 'incident_date' in X.columns:
    X['incident_year'] = pd.to_datetime(X['incident_date']).dt.year
    X['incident_month'] = pd.to_datetime(X['incident_date']).dt.month
    X = X.drop('incident_date', axis=1)

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train XGBoost model with class imbalance handling
print("\nTraining XGBoost model...")

model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle imbalance
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluate the model
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print(f"\n=== ROC-AUC Score ===")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Save the model and encoders
print("\nSaving model...")
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("\n✅ Model trained and saved successfully!")
print("Files created:")
print("  - fraud_model.pkl (the trained model)")
print("  - label_encoders.pkl (for encoding categorical data)")
print("  - feature_names.pkl (column names)")