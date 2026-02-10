import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Load the trained model and encoders
@st.cache_resource
def load_model():
    with open('fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, encoders, features

model, label_encoders, feature_names = load_model()

# App title and description
st.set_page_config(page_title="Insurance Fraud Detection", page_icon="🔍", layout="wide")

st.title("🔍 Insurance Fraud Detection System")
st.markdown("Upload insurance claim data to check for potential fraud")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This AI-powered system analyzes insurance claims to detect potential fraud. "
    "Upload a CSV/Excel file with claim details or fill in the form manually."
)

# Main content
tab1, tab2 = st.tabs(["📄 Upload File", "✍️ Manual Entry"])

# Tab 1: File Upload
with tab1:
    st.header("Upload Claims File")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! ({len(df)} claims)")
            
            # Show preview
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head())
            
            if st.button("🔍 Analyze All Claims", type="primary"):
                with st.spinner("Analyzing claims..."):
                    # Prepare data
                    X = df.copy()
                    
                    # Remove target if present
                    if 'fraud_reported' in X.columns:
                        X = X.drop('fraud_reported', axis=1)
                    
                    # Encode categorical columns
                    for col in label_encoders.keys():
                        if col in X.columns:
                            le = label_encoders[col]
                            X[col] = X[col].astype(str)
                            # Handle unseen categories
                            X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                            X[col] = le.transform(X[col])
                    
                    # Handle dates
                    if 'policy_bind_date' in X.columns:
                        X['policy_bind_year'] = pd.to_datetime(X['policy_bind_date']).dt.year
                        X['policy_bind_month'] = pd.to_datetime(X['policy_bind_date']).dt.month
                        X = X.drop('policy_bind_date', axis=1)
                    
                    if 'incident_date' in X.columns:
                        X['incident_year'] = pd.to_datetime(X['incident_date']).dt.year
                        X['incident_month'] = pd.to_datetime(X['incident_date']).dt.month
                        X = X.drop('incident_date', axis=1)
                    
                    # Ensure correct column order
                    X = X[feature_names]
                    
                    # Predict
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)[:, 1]
                    
                    # Add results to dataframe
                    df['Fraud_Prediction'] = ['🚨 FRAUDULENT' if p == 1 else '✅ LEGITIMATE' for p in predictions]
                    df['Fraud_Probability'] = [f"{p*100:.1f}%" for p in probabilities]
                    
                    # Display results
                    st.success("Analysis complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Claims", len(df))
                    with col2:
                        fraud_count = sum(predictions)
                        st.metric("Fraudulent Claims", fraud_count)
                    with col3:
                        legit_count = len(predictions) - fraud_count
                        st.metric("Legitimate Claims", legit_count)
                    
                    # Show flagged claims
                    st.subheader("🚨 Flagged Claims (High Risk)")
                    flagged = df[df['Fraud_Prediction'] == '🚨 FRAUDULENT']
                    if len(flagged) > 0:
                        st.dataframe(flagged[['policy_number', 'total_claim_amount', 'Fraud_Prediction', 'Fraud_Probability']])
                    else:
                        st.info("No fraudulent claims detected!")
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="fraud_analysis_results.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Tab 2: Manual Entry
with tab2:
    st.header("Enter Claim Details Manually")
    st.info("Fill in the details below to check a single claim")
    
    col1, col2 = st.columns(2)
    
    with col1:
        months_as_customer = st.number_input("Months as Customer", min_value=0, value=100)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        policy_state = st.selectbox("Policy State", ['OH', 'IN', 'IL'])
        policy_deductable = st.number_input("Policy Deductible ($)", min_value=0, value=1000)
        policy_annual_premium = st.number_input("Annual Premium ($)", min_value=0.0, value=1500.0)
        insured_sex = st.selectbox("Insured Sex", ['MALE', 'FEMALE'])
        insured_education_level = st.selectbox("Education Level", ['MD', 'PhD', 'Associate', 'Masters', 'High School', 'College', 'JD'])
    
    with col2:
        incident_type = st.selectbox("Incident Type", ['Single Vehicle Collision', 'Vehicle Theft', 'Multi-vehicle Collision', 'Parked Car'])
        collision_type = st.selectbox("Collision Type", ['Side Collision', 'Rear Collision', 'Front Collision', '?'])
        incident_severity = st.selectbox("Incident Severity", ['Major Damage', 'Minor Damage', 'Total Loss', 'Trivial Damage'])
        authorities_contacted = st.selectbox("Authorities Contacted", ['Police', 'Fire', 'Ambulance', 'Other', 'None'])
        total_claim_amount = st.number_input("Total Claim Amount ($)", min_value=0, value=5000)
        injury_claim = st.number_input("Injury Claim ($)", min_value=0, value=0)
        property_claim = st.number_input("Property Claim ($)", min_value=0, value=0)
    
    if st.button("🔍 Check for Fraud", type="primary"):
        # This is simplified - in production you'd need to fill all 39 features
        st.warning("⚠️ Manual entry requires all 39 features. Use file upload for complete analysis.")
        st.info("For now, use the 'Upload File' tab with a complete dataset.")

# Footer
st.markdown("---")
st.markdown("**Fraud Detection System** | Powered by XGBoost & Streamlit")