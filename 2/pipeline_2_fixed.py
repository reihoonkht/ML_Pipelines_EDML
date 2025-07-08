import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# ... path setup ...

# Load data
raw_data_file = os.path.join(project_root, "datasets", "second_dataset", "second_dataset.csv")
raw_data = pd.read_csv(raw_data_file)

# Select meaningful features (remove data leakage)
feature_columns = ['sex', 'age', 'c_charge_degree', 'race', 'priors_count', 'days_b_screening_arrest']
target_column = 'score_text'

# Extract features and target BEFORE any filtering
X = raw_data[feature_columns].copy()
y = raw_data[target_column].copy()

print(f"Original dataset: {len(X)} records")

# FIXED: Train-test split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# FIXED: Apply filtering AFTER split and only to training data if needed
# Document the rationale for filtering
def apply_data_quality_filters(X, y, filter_name=""):
    """Apply data quality filters with clear documentation"""
    original_size = len(X)
    
    # Filter 1: Remove extreme values in days_b_screening_arrest
    # Rationale: Values beyond ±30 days may indicate data quality issues
    mask = (X['days_b_screening_arrest'] >= -30) & (X['days_b_screening_arrest'] <= 30)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    removed = original_size - len(X_filtered)
    print(f"{filter_name} - Removed {removed} records ({removed/original_size*100:.1f}%) due to extreme days_b_screening_arrest")
    
    return X_filtered, y_filtered

# Apply filtering to training data only
X_train_filtered, y_train_filtered = apply_data_quality_filters(X_train, y_train, "Training")

# For test data, decide whether to filter or keep all data
# Option 1: Filter test data the same way (recommended for consistency)
X_test_filtered, y_test_filtered = apply_data_quality_filters(X_test, y_test, "Test")

# Option 2: Keep all test data to evaluate on full distribution
# X_test_filtered, y_test_filtered = X_test, y_test

# FIXED: Keep original 3-class problem (remove representational bias)
# y_train_filtered = y_train_filtered.replace('Medium', 'Low')  # REMOVED
# y_test_filtered = y_test_filtered.replace('Medium', 'Low')    # REMOVED

print(f"Training data after filtering: {len(X_train_filtered)} records")
print(f"Test data after filtering: {len(X_test_filtered)} records")

print("\nTarget distribution in training data:")
print(y_train_filtered.value_counts())

# Define preprocessing
categorical_features = ['sex', 'c_charge_degree', 'race']
numeric_features = ['age', 'priors_count', 'days_b_screening_arrest']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features)
    ]
)

# Create pipeline (NO data leakage)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr'))
])

# Fit only on training data
pipeline.fit(X_train_filtered, y_train_filtered)

# Evaluate
train_score = pipeline.score(X_train_filtered, y_train_filtered)
test_score = pipeline.score(X_test_filtered, y_test_filtered)

print(f"\nTraining accuracy: {train_score:.3f}")
print(f"Testing accuracy: {test_score:.3f}")

y_pred = pipeline.predict(X_test_filtered)
print("\nClassification Report:")
print(classification_report(y_test_filtered, y_pred, zero_division=0))

# Fairness check
print("\nFairness Check - Predictions by Race:")
test_results = X_test_filtered.copy()
test_results['true_label'] = y_test_filtered
test_results['predicted_label'] = y_pred
fairness_check = pd.crosstab(test_results['race'], test_results['predicted_label'], normalize='index') * 100
print(fairness_check.round(1))                       
