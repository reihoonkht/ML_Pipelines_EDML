# BIAS-AWARE VERSION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load data
raw_data = pd.read_csv(raw_data_file)

# REMOVED: Biased transformation that misrepresents Medium risk as Low
# raw_data = raw_data.replace('Medium', "Low")  # BIAS SOURCE - REMOVED

# Use meaningful features (remove data leakage)
feature_columns = ['sex', 'age', 'c_charge_degree', 'race', 'priors_count', 'days_b_screening_arrest']
X = raw_data[feature_columns].copy()
y = raw_data['score_text'].copy()  # Keep original 3-class representation

# Train-test split with stratification to maintain fair representation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipeline
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

# Multi-class classification preserving all risk categories
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr'))
])

pipeline.fit(X_train, y_train)

# Evaluate with fairness in mind
y_pred = pipeline.predict(X_test)
print("Classification Report (Fair Representation):")
print(classification_report(y_test, y_pred, zero_division=0))

# Check for demographic parity
print("\nFairness Check - Predictions by Race:")
test_results = X_test.copy()
test_results['true_label'] = y_test
test_results['predicted_label'] = y_pred
fairness_check = pd.crosstab(test_results['race'], test_results['predicted_label'], normalize='index') * 100
print(fairness_check.round(1))                       
