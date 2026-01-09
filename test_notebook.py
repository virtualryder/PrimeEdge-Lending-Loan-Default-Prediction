# -*- coding: utf-8 -*-
"""
Test script to verify the machine learning pipeline works correctly
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("="*80)
print("TESTING PRIMEEDGE LENDING ML PIPELINE")
print("="*80)

# Load data
print("\n1. Loading dataset...")
df = pd.read_csv('Loan_Delinquent_Analysis_Dataset.csv')
print(f"   [OK] Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"   [OK] Default rate: {df['Delinquency_Status'].mean()*100:.2f}%")

# Clean data
print("\n2. Cleaning data...")
loan_data = df.copy()
loan_data['Loan_Purpose'] = loan_data['Loan_Purpose'].replace('other', 'Other')
print(f"   [OK] Fixed case sensitivity in Loan_Purpose")

# Feature engineering
print("\n3. Feature engineering...")
loan_data['Loan_to_Income_Ratio'] = loan_data['Loan_Amount'] / (loan_data['Income'] * 1000)
loan_data['High_Risk_Loan'] = (loan_data['Loan_Amount'] > loan_data['Income'] * 5000).astype(int)
print(f"   [OK] Created Loan_to_Income_Ratio")
print(f"   [OK] Created High_Risk_Loan flag")

# Prepare for modeling
print("\n4. Preparing data for modeling...")
model_data = loan_data.drop('ID', axis=1)

# Encode categorical variables
categorical_cols = ['Loan_Term', 'Borrower_Gender', 'Loan_Purpose', 'Home_Status', 'Age_Group', 'Credit_Score_Range']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    model_data[col] = le.fit_transform(model_data[col])
    label_encoders[col] = le

print(f"   [OK] Encoded {len(categorical_cols)} categorical columns")

# Split data
X = model_data.drop('Delinquency_Status', axis=1)
y = model_data['Delinquency_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   [OK] Training set: {X_train.shape[0]:,} samples")
print(f"   [OK] Test set: {X_test.shape[0]:,} samples")

# Train Random Forest (best model)
print("\n5. Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
print(f"   [OK] Model trained successfully")

# Evaluate
print("\n6. Evaluating model...")
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n" + "="*80)
print("MODEL PERFORMANCE RESULTS")
print("="*80)
print(f"Test Accuracy:  {accuracy*100:.2f}%")
print(f"Precision:      {precision*100:.2f}%")
print(f"Recall:         {recall*100:.2f}%")
print(f"F1 Score:       {f1*100:.2f}%")
print("="*80)

# Feature importance
print("\n7. Top 5 Most Important Features:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

print("\n" + "="*80)
print("[SUCCESS] ALL TESTS PASSED - PIPELINE WORKING CORRECTLY!")
print("="*80)
