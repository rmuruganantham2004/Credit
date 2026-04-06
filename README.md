# End-to-End Credit Risk Machine Learning System

This repository contains a full end-to-end Machine Learning system that predicts credit risk (probability of loan default) from the famous **German Credit Dataset**. The project focuses not only on predictive performance but also integrates **Model Transparency**, **Interpretability (Explainable AI)**, and **Algorithmic Fairness**.

It includes a fully-functional Web App built using **FastAPI** with a modern dynamic dashboard to test the model intuitively.

---

## 🌟 Key Features

1. **Robust Predictive Modeling**: Tested across Logistic Regression and Random Forest classifiers utilizing structured `scikit-learn` Column Transformer pipelines.
2. **Bias Mitigation & Fairness Check**: Utilized **Fairlearn** metric tools to measure bias (Demographic Parity Difference and Equal Opportunity Difference) against sensitive attributes like *Gender* and *Age*. Applied the `ThresholdOptimizer` to reduce demographic disparity natively!
3. **Deep Explainability**: Integrated **SHAP** (Shapley Additive Explanations) and **LIME** local HTML explanations to decompose exactly what features trigger specific predictive decisions.
4. **Rich Web Application**: A fully baked FastAPI backend integrated with a stunning, dynamically colored HTML frontend dashboard (employing "Glassmorphism" features) that triggers ML models locally.

---

## 🏗 Project Architecture

* `src/data_pipeline.py`: Automatically downloads the `OpenML German Credit` dataset, defines schemas, engineered novel attributes (like `amount_per_duration`), limits sensitive exposure, and performs categorical One-Hot representation.
* `src/eda.py`: Employs Matplotlib and Seaborn to automatically output insights natively to the `/static/eda` paths. 
* `src/model_trainer.py`: Compares algorithms dynamically, logs AUC/F1-score improvements, measures disparity drops, structures a mitigation pipeline, and drops serialization `*.pkl` artifacts into `/models`.
* `src/explainability.py`: Parses the model output directly through SHAP Tree Explainers / Linear Explainers to generate live global/local charts in `/static/explain/`.
* `app.py`: FastAPI server endpoint routing web sessions and bridging static image folders.

---

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python 3.9+ (Python 3.10 is recommended).

### 1. Installation 
Clone the repository and install all required dependencies using `pip`.
```bash
git clone https://github.com/rmuruganantham2004/Credit.git
cd Credit
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. (Optional) Rebuilding the Models
If you'd like to reconstruct the data splits, visualize the EDA, or natively retrain the baseline algorithms:
```bash
python src/data_pipeline.py
python src/eda.py
python src/model_trainer.py
python src/explainability.py
```
*Note: Make sure this is run sequentially so that `app.py` recognizes your updated `/models/*.pkl` objects.*

### 3. Launch the Server!
Run the backend web application via Uvicorn:
```bash
uvicorn app:app --reload
```
Navigate to **http://127.0.0.1:8000/** to access the Fair AI applicant dashboard!

---

## 🤔 Technical Design Decisions
- **Missing XGBoost**: XGBoost was tested initially but omitted gracefully from the final iteration due to `libomp` binary compatibility requirements for pure MacOS deployment. Random Forest natively provides tree-based predictive stability.
- **Why FastAPI?**: Used over Streamlit/Flask to natively stream background asynchronous ML computation dynamically over Jinja templates without bogging down server concurrency operations.

## 📄 License
This project falls under open-source MIT guidelines.
