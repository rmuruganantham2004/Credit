from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np
import shap
import os
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

os.makedirs("static/explain", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

preprocessor, model = None, None

def load_models():
    global preprocessor, model
    try:
        if os.path.exists('models/preprocessor.pkl') and os.path.exists('models/best_model.pkl'):
            preprocessor = joblib.load('models/preprocessor.pkl')
            model = joblib.load('models/best_model.pkl')
            logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Models not loaded. Error: {e}")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request,
                  duration: float = Form(24),
                  credit_history: str = Form('existing paid'),
                  credit_amount: float = Form(3000),
                  employment: str = Form('1<=X<4'),
                  personal_status: str = Form('male single'),
                  age: float = Form(35),
                  job: str = Form('skilled')):
                  
    if not preprocessor or not model:
        load_models()
        if not preprocessor:
            return templates.TemplateResponse("index.html", {"request": request, "error": "Model not trained yet."})

    # For simplicity, filling unprovided fields with common values
    # To properly predict German Credit dataset we need these 20 fields.
    # We populate the rest with constants so predict can work
    input_data = {
        "checking_status": "no checking",
        "duration": duration,
        "credit_history": credit_history,
        "purpose": "radio/tv",
        "credit_amount": credit_amount,
        "savings_status": "no known savings",
        "employment": employment,
        "installment_commitment": 4.0,
        "personal_status": personal_status,
        "other_parties": "none",
        "residence_since": 4.0,
        "property_magnitude": "real estate",
        "age": age,
        "other_payment_plans": "none",
        "housing": "own",
        "existing_credits": 1.0,
        "job": job,
        "num_dependents": 1.0,
        "own_telephone": "none",
        "foreign_worker": "yes"
    }
    
    df = pd.DataFrame([input_data])
    
    df['gender'] = df['personal_status'].apply(lambda x: 'female' if 'female' in str(x).lower() else 'male')
    df['age_group'] = df['age'].apply(lambda a: 'youth' if a < 25 else ('senior' if a > 60 else 'adult'))
    df['amount_per_duration'] = df['credit_amount'] / (df['duration'] + 1e-5)
    
    X_transformed = preprocessor.transform(df)
    
    pred = model.predict(X_transformed)[0]
    prob = model.predict_proba(X_transformed)[0][1]
    
    risk = "High Risk (Default Expected)" if pred == 1 else "Low Risk (Good Credit)"
    color = "#e74c3c" if pred == 1 else "#2ecc71"
    
    num_features = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_features = df.select_dtypes(include=['category', 'object']).columns.tolist()
    try:
        ohe_cols = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_features)
        feature_names = num_features + list(ohe_cols)
    except:
        feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
    
    # SHAP Generation Live
    model_type = type(model).__name__
    plt.figure(figsize=(8,3))    
    if model_type in ['RandomForestClassifier', 'XGBClassifier', 'DecisionTreeClassifier']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        ev = explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value
    else:
        try:
            # Linear Explainer requires background data, which we don't hold in memory
            # So we use a generic bar SHAP plot if it's LR just taking coefficients? 
            # Better: avoid breaking
            pass 
        except:
            pass
            
    if 'sv' in locals():
        shap.decision_plot(ev, sv[0], features=X_transformed[0], feature_names=feature_names, show=False)
        explain_path = f"static/explain/shap_sample_live_{int(age)}_{int(credit_amount)}.png"
        plt.savefig(explain_path, bbox_inches='tight')
        plt.close()
    else:
        explain_path = None

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": risk,
        "pred_color": color,
        "probability": f"{prob*100:.1f}%",
        "explain_img": f"/{explain_path}" if explain_path else None
    })
