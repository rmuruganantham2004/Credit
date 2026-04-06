import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fetch_and_preprocess_german_credit():
    """
    Fetches the German Credit dataset from OpenML and prepares it for modeling.
    Returns the raw features, target, preprocessing pipeline, and feature lists.
    """
    logger.info("Fetching German Credit data from OpenML...")
    # data_id = 31 corresponds to German Credit
    data = fetch_openml(data_id=31, as_frame=True, parser='auto')
    
    X = data.data
    y = data.target
    
    # OpenML class target: 'good' or 'bad'. Let's predict Default probability (1 = Default/bad, 0 = No Default/good)
    y = y.map({'good': 0, 'bad': 1})
    
    # Simplify the sensitive attribute `personal_status` into `gender`
    if 'personal_status' in X.columns:
        X['gender'] = X['personal_status'].apply(lambda x: 'female' if 'female' in str(x).lower() else 'male')
    
    # Categorize age
    if 'age' in X.columns:
        X['age_group'] = X['age'].apply(lambda a: 'youth' if a < 25 else ('senior' if a > 60 else 'adult'))
        
    # Feature Engineering Example: Value per unit of time
    if 'credit_amount' in X.columns and 'duration' in X.columns:
        X['amount_per_duration'] = X['credit_amount'] / (X['duration'] + 1e-5)
    
    # Identify column types
    cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
    num_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    # Create sklearn preprocessing pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    
    logger.info(f"Numeric features: {len(num_features)}, Categorical features: {len(cat_features)}")
    
    return X, y, preprocessor, cat_features, num_features

def get_train_test_splits():
    """
    Returns train and test splits, retaining raw data formats suitable for both
    Fairlearn bias analysis and scikit-learn models through pipelines.
    """
    X, y, preprocessor, cat_features, num_features = fetch_and_preprocess_german_credit()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, preprocessor, cat_features, num_features

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preproc, cat_cols, num_cols = get_train_test_splits()
    logger.info("Data pipeline verified successfully.")
