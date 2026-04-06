import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from data_pipeline import fetch_and_preprocess_german_credit

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_eda():
    """Generates Exploratory Data Analysis plots and saves them."""
    X, y, _, _, _ = fetch_and_preprocess_german_credit()
    df = X.copy()
    df['default'] = y
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'static/eda')
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Target Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='default', palette='Set2')
    plt.title('Distribution of Target (0: No Default, 1: Default)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'target_distribution.png'))
    plt.close()
    
    # 2. Gender vs Default
    if 'gender' in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(data=df, x='gender', hue='default', palette='Set1')
        plt.title('Default Rates by Gender')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gender_distribution.png'))
        plt.close()
        
    # 3. Age Group vs Default
    if 'age_group' in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(data=df, x='age_group', hue='default', palette='viridis')
        plt.title('Default Rates by Age Group')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'age_distribution.png'))
        plt.close()
        
    # 4. Correlation Heatmap
    plt.figure(figsize=(10,8))
    numeric_df = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Numeric Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'))
    plt.close()
    
    logger.info(f"EDA visualizations saved to {save_dir}/")

if __name__ == "__main__":
    # We will run this from the src/ directory or root directory.
    # We should adjust imports if run as a script. By using `python src/eda.py`, sys.path might need an update.
    # To be safe:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    run_eda()
