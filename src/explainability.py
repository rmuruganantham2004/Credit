import os
import joblib
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import logging
from data_pipeline import get_train_test_splits

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_explanations(sample_idx=0):
    """Generates SHAP and LIME explanations for instances and globally."""
    logger.info("Loading preprocessor and model...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'models')
    
    try:
        preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
        model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    X_train, X_test, y_train, y_test, _, cat_features, num_features = get_train_test_splits()
    
    idx = sample_idx
    sample_raw = X_test.iloc[[idx]]
    y_true = y_test.iloc[idx]
    
    logger.info("Transforming data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    sample_transformed = preprocessor.transform(sample_raw)
    
    try:
        ohe_cols = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_features)
        feature_names = num_features + list(ohe_cols)
    except:
        feature_names = [f"Feature_{i}" for i in range(X_train_transformed.shape[1])]
    
    model_type = type(model).__name__
    save_dir = os.path.join(base_dir, 'static/explain')
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("Generating SHAP local explanation...")
    plt.figure()
    if model_type in ['RandomForestClassifier', 'XGBClassifier', 'DecisionTreeClassifier']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_transformed)
        if isinstance(shap_values, list):
            sv = shap_values[1]
            ev = explainer.expected_value[1]
        else:
            sv = shap_values
            ev = explainer.expected_value
        
        shap.decision_plot(ev, sv[0], features=sample_transformed[0], feature_names=feature_names, show=False)
    else:
        explainer = shap.LinearExplainer(model, X_train_transformed)
        shap_values = explainer.shap_values(sample_transformed)
        sv = shap_values
        ev = explainer.expected_value
        shap.summary_plot(sv, sample_transformed, feature_names=feature_names, show=False)
        
    plt.savefig(os.path.join(save_dir, 'shap_sample.png'), bbox_inches='tight')
    plt.close()
    
    logger.info("Generating SHAP global explanation...")
    plt.figure()
    if model_type in ['RandomForestClassifier', 'XGBClassifier']:
        shap_values_global = explainer.shap_values(X_test_transformed[:200]) 
        sv_g = shap_values_global[1] if isinstance(shap_values_global, list) else shap_values_global
        shap.summary_plot(sv_g, X_test_transformed[:200], feature_names=feature_names, show=False)
        plt.savefig(os.path.join(save_dir, 'shap_global.png'), bbox_inches='tight')
        plt.close()
        
    logger.info("Generating LIME explanation...")
    # Use toarray() if LIME struggles with sparse matrices, but pipeline outputs ndarray due to sparse_output=False
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_transformed,
        feature_names=feature_names,
        class_names=["No Default", "Default"],
        mode='classification'
    )
    
    exp = lime_explainer.explain_instance(
        data_row=sample_transformed[0], 
        predict_fn=model.predict_proba
    )
    
    exp.save_to_file(os.path.join(save_dir, 'lime_sample.html'))
    logger.info(f"Local explanations generated for test index {sample_idx} (True Class: {y_true}). Exlanations saved to {save_dir}/")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    generate_explanations()
