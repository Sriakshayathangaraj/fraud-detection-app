import pandas as pd

# Load the dataset
df = pd.read_excel('insurance_claims.xlsx')

# Display basic info
print("Dataset loaded successfully!")
print(f"\nTotal records: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Fraud Distribution ===")
print(df['fraud_reported'].value_counts())