import os
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer

from data_pipeline import get_train_test_splits

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def train_and_evaluate():
    """Trains baseline models, evaluates fairness, mitigates bias, and saves models."""
    X_train, X_test, y_train, y_test, preprocessor, cat_features, num_features = get_train_test_splits()
    
    sensitive_train = X_train['gender']
    sensitive_test = X_test['gender']
    
    logger.info("Transforming data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_auc = 0
    best_model_name = ""
    best_model = None
    
    logger.info("Training baseline models...")
    for name, model in models.items():
        model.fit(X_train_transformed, y_train)
        y_pred = model.predict(X_test_transformed)
        y_prob = model.predict_proba(X_test_transformed)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        logger.info(f"{name} -> AUC: {auc:.4f}, F1: {f1:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = model
            
    logger.info(f"Best baseline model: {best_model_name} (AUC: {best_auc:.4f})")
    
    # Fairness Analysis on best baseline model
    y_pred_best = best_model.predict(X_test_transformed)
    dp_diff = demographic_parity_difference(y_test, y_pred_best, sensitive_features=sensitive_test)
    eo_diff = equalized_odds_difference(y_test, y_pred_best, sensitive_features=sensitive_test)
    logger.info(f"Baseline Fairness - Demo Parity Diff: {dp_diff:.4f}, Equal Opp Diff: {eo_diff:.4f}")
    
    # Bias Mitigation using ThresholdOptimizer
    logger.info("Applying ThresholdOptimizer to mitigate bias (Demographic Parity)...")
    mitigator = ThresholdOptimizer(
        estimator=best_model,
        constraints="demographic_parity",
        predict_method="predict_proba",
        prefit=True
    )
    
    mitigator.fit(X_train_transformed, y_train, sensitive_features=sensitive_train)
    
    y_pred_mitigated = mitigator.predict(X_test_transformed, sensitive_features=sensitive_test)
    
    acc_mit = accuracy_score(y_test, y_pred_mitigated)
    f1_mit = f1_score(y_test, y_pred_mitigated)
    dp_diff_mit = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=sensitive_test)
    eo_diff_mit = equalized_odds_difference(y_test, y_pred_mitigated, sensitive_features=sensitive_test)
    
    logger.info(f"Mitigated Model - Accuracy: {acc_mit:.4f}, F1: {f1_mit:.4f}")
    logger.info(f"Mitigated Fairness - Demo Parity Diff: {dp_diff_mit:.4f}, Equal Opp Diff: {eo_diff_mit:.4f}")
    
    # Save models and preprocessor
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(preprocessor, f'{save_dir}/preprocessor.pkl')
    joblib.dump(best_model, f'{save_dir}/best_model.pkl')
    joblib.dump(mitigator, f'{save_dir}/mitigator.pkl')
    logger.info(f"Models and preprocessor saved to {save_dir}/")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    train_and_evaluate()
