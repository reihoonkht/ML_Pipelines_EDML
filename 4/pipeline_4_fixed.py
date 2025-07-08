import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

project_root = get_project_root()

# FIXED: Add error handling for data loading
try:
    raw_data_file = os.path.join(project_root, "datasets", "first_dataset", "first_dataset.csv")
    data = pd.read_csv(raw_data_file)
    
    if data.empty:
        raise ValueError("Dataset is empty")
    
    print(f"Loaded dataset with {len(data)} records and {len(data.columns)} features")
    
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# FIXED: Validate target variable exists
if 'salary' not in data.columns:
    print("Error: 'salary' column not found in dataset")
    sys.exit(1)

# Prepare features and target
X = data.drop('salary', axis=1)
y = data['salary']

print(f"Target distribution:\n{y.value_counts()}")

# FIXED: More robust feature type detection
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
print(f"Numerical features ({len(numerical_features)}): {numerical_features}")

# FIXED: Proper preprocessing with scaling and imputation
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())  # FIXED: Add scaling for numerical features
        ]), numerical_features)
    ],
    remainder='drop'  
)

# FIXED: Proper pipeline definition
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42, n_estimators=100))  # FIXED: Add random_state
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train model
try:
    pipeline.fit(X_train, y_train)
    print("Model training completed successfully")
except Exception as e:
    print(f"Error during training: {e}")
    sys.exit(1)

# Evaluate on clean data
y_pred = pipeline.predict(X_test)
accuracy_before_noise = accuracy_score(y_test, y_pred)

print(f"\n=== CLEAN DATA PERFORMANCE ===")
print(f"Accuracy: {accuracy_before_noise:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# FIXED: Principled robustness testing
def add_scaled_noise(X_test, noise_levels=[0.1, 0.2, 0.5], random_seed=42):
    """
    Add scaled noise to numerical features for robustness testing.
    
    Args:
        X_test: Test data
        noise_levels: List of noise levels as fraction of feature std
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary of noisy datasets
    """
    np.random.seed(random_seed)
    
    # Get numerical columns that actually exist in the data
    numerical_cols = X_test.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numerical_cols) == 0:
        print("Warning: No numerical columns found for noise addition")
        return {}
    
    # Calculate feature-specific noise scales
    feature_stds = X_test[numerical_cols].std()
    
    noisy_datasets = {}
    
    for noise_level in noise_levels:
        X_noisy = X_test.copy()
        
        # Add proportional noise to each numerical feature
        for col in numerical_cols:
            noise_std = feature_stds[col] * noise_level
            noise = np.random.normal(0, noise_std, len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise
            
            # FIXED: Ensure realistic values (e.g., age can't be negative)
            if col == 'age':
                X_noisy[col] = np.clip(X_noisy[col], 0, 120)
            elif col in ['capital-gain', 'capital-loss']:
                X_noisy[col] = np.clip(X_noisy[col], 0, None)  # Can't be negative
            elif col == 'hours-per-week':
                X_noisy[col] = np.clip(X_noisy[col], 0, 168)  # Max hours in a week
        
        noisy_datasets[noise_level] = X_noisy
    
    return noisy_datasets

# Test robustness with multiple noise levels
print(f"\n=== ROBUSTNESS TESTING ===")
noise_levels = [0.05, 0.1, 0.2, 0.3]  # 5%, 10%, 20%, 30% of feature std
noisy_datasets = add_scaled_noise(X_test, noise_levels)

robustness_results = []

for noise_level, X_noisy in noisy_datasets.items():
    try:
        y_pred_noisy = pipeline.predict(X_noisy)
        accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
        accuracy_drop = accuracy_before_noise - accuracy_noisy
        
        robustness_results.append({
            'noise_level': noise_level,
            'accuracy': accuracy_noisy,
            'accuracy_drop': accuracy_drop,
            'relative_drop': accuracy_drop / accuracy_before_noise * 100
        })
        
        print(f"Noise level {noise_level*100:4.1f}%: Accuracy = {accuracy_noisy:.4f} "
              f"(drop: {accuracy_drop:.4f}, {accuracy_drop/accuracy_before_noise*100:.1f}%)")
              
    except Exception as e:
        print(f"Error with noise level {noise_level}: {e}")

# Summary
print(f"\n=== ROBUSTNESS SUMMARY ===")
print(f"Baseline accuracy: {accuracy_before_noise:.4f}")

if robustness_results:
    max_drop = max(result['relative_drop'] for result in robustness_results)
    print(f"Maximum relative accuracy drop: {max_drop:.1f}%")
    
    if max_drop < 5:
        print("✅ Model shows good robustness to noise")
    elif max_drop < 15:
        print("⚠️  Model shows moderate robustness to noise")
    else:
        print("❌ Model shows poor robustness to noise")
else:
    print("❌ Robustness testing failed")
