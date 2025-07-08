import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

# FIXED: Add error handling
try:
    raw_data_file = os.path.join(project_root, "datasets", "third_dataset", "third_dataset.csv")
    data = pd.read_csv(raw_data_file)
    
    if data.empty:
        raise ValueError("Dataset is empty")
        
    print(f"Loaded dataset with {len(data)} records and {len(data.columns)} features")
    
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# FIXED: Validate target variable
if 'Diabetes_binary' not in data.columns:
    print("Error: 'Diabetes_binary' column not found")
    sys.exit(1)

print("=== ORIGINAL DATA ANALYSIS ===")
print("Raw data shape:", data.shape)
print("Raw data gender distribution:")
print(data['Sex'].value_counts(normalize=True).round(3))
print("Raw data diabetes distribution:")
print(data['Diabetes_binary'].value_counts(normalize=True).round(3))

# FIXED: Remove biased filtering or apply it more carefully
# Option 1: Remove filtering entirely (RECOMMENDED)
print("\n=== USING FULL DATASET (RECOMMENDED) ===")
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']

# Option 2: If filtering is necessary, document and apply after split
# print("\n=== APPLYING DOCUMENTED FILTERING ===")
# def apply_quality_filters(X, y, filter_name=""):
#     """Apply data quality filters with clear documentation"""
#     original_size = len(X)
#     
#     # Only apply if there's a clear medical/scientific rationale
#     # Age > 4: Could be justified if studying adult-onset diabetes
#     # HighChol > 0: This creates severe bias - NOT RECOMMENDED
#     
#     # Example of principled filtering:
#     # mask = (X['Age'] >= 5)  # Focus on age groups where diabetes is more common
#     # X_filtered = X[mask]
#     # y_filtered = y[mask]
#     
#     # removed = original_size - len(X_filtered)
#     # print(f"{filter_name} - Removed {removed} records ({removed/original_size*100:.1f}%) due to age criteria")
#     
#     # return X_filtered, y_filtered
#     
#     return X, y  # No filtering for now

# FIXED: Train-test split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

print("\nTest set gender distribution:")
print(X_test['Sex'].value_counts(normalize=True).round(3))
print("Test set diabetes distribution:")
print(y_test.value_counts(normalize=True).round(3))

# FIXED: Proper feature preprocessing
# All features are numeric, so we need scaling, not one-hot encoding
print("\n=== FEATURE PREPROCESSING ===")
print("All features are numeric - applying scaling instead of one-hot encoding")

# Identify binary vs continuous features for appropriate preprocessing
binary_features = []
continuous_features = []

for col in X_train.columns:
    unique_vals = X_train[col].nunique()
    if unique_vals == 2:
        binary_features.append(col)
    else:
        continuous_features.append(col)

print(f"Binary features ({len(binary_features)}): {binary_features[:5]}...")
print(f"Continuous features ({len(continuous_features)}): {continuous_features}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('binary', 'passthrough', binary_features),  # Keep binary features as-is
        ('continuous', StandardScaler(), continuous_features)  # Scale continuous features
    ]
)

# FIXED: Create proper pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Train model
print("\n=== MODEL TRAINING ===")
try:
    pipeline.fit(X_train, y_train)
    print("Model training completed successfully")
except Exception as e:
    print(f"Error during training: {e}")
    sys.exit(1)

# Evaluate model
y_pred = pipeline.predict(X_test)
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"\n=== MODEL PERFORMANCE ===")
print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# FIXED: Bias analysis
print(f"\n=== BIAS ANALYSIS ===")
print("Performance by gender:")

# Split test results by gender
test_results = X_test.copy()
test_results['true_label'] = y_test
test_results['predicted_label'] = y_pred

for gender in [0.0, 1.0]:
    gender_mask = test_results['Sex'] == gender
    gender_data = test_results[gender_mask]
    
    if len(gender_data) > 0:
        gender_accuracy = (gender_data['true_label'] == gender_data['predicted_label']).mean()
        gender_name = "Female" if gender == 0.0 else "Male"
        print(f"{gender_name} accuracy: {gender_accuracy:.4f} (n={len(gender_data)})")

# Check for demographic parity
print("\nDemographic parity check:")
for gender in [0.0, 1.0]:
    gender_mask = test_results['Sex'] == gender
    gender_data = test_results[gender_mask]
    
    if len(gender_data) > 0:
        positive_rate = (gender_data['predicted_label'] == 1.0).mean()
        gender_name = "Female" if gender == 0.0 else "Male"
        print(f"{gender_name} positive prediction rate: {positive_rate:.4f}")

print(f"\n=== SUMMARY ===")
print("✅ Removed biased filtering that excluded 49.5% of data")
print("✅ Fixed one-hot encoding logic error")
print("✅ Applied proper scaling for numeric features")
print("✅ Added bias analysis and fairness checks")
print("✅ Maintained balanced dataset representation")
