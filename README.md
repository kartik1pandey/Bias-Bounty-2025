# Bias Bounty Competition: Loan Approval Pipeline

## Overview
This repository contains a machine learning pipeline for the Bias Bounty competition, designed to predict loan approvals while identifying and mitigating biases in the `loan_access_dataset.csv`. The pipeline uses Logistic Regression and XGBoost with 5-fold cross-validation, achieving validation accuracies of 0.6284 (Logistic) and 0.6200 (XGBoost). It employs `fairlearn` for fairness auditing (e.g., Gender DPD: 0.4167, Non-binary recall: 0.3125), `ExponentiatedGradient` for bias mitigation, and visualizations (SHAP, bias-variance, fairness metrics) for interpretability. The code is production-ready with logging, error handling, and a detailed AI Risk Report.

## Installation

### Prerequisites
- Python 3.8+
- Dependencies:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost fairlearn shap imbalanced-learn joblib scipy
  ```

### Dataset
- Place `loan_access_dataset.csv` and `test.csv` in `/kaggle/input/bias-bounty/` or adjust the file paths in `loan_model.py`.

## Usage

### Running the Pipeline
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Run the pipeline:
   ```bash
   python loan_model.py
   ```
3. Outputs:
   - **Submission**: `submission_5fold_xgb_YYYY-MM-DD-HH-MM.csv` (predictions for `test.csv`).
   - **Visualizations** (in `charts/`):
     - `approval_rates_gender.png`, `approval_rates_race.png`, `approval_rates_zip_code_group.png`: Approval rates with confidence intervals.
     - `shap_importance.png`: SHAP feature importance.
     - `fairness_metrics.png`: DPD and EOD by attribute.
     - `bias_variance.png`: Bias-variance trade-off.
     - `bias_visualization.png`: Gender-Race approval rate heatmap.
   - **Report**: `ai_risk_report.md` (detailed fairness analysis).
   - **Console**: Cross-validation accuracies, fairness metrics (DPD, EOD), and bias-variance metrics.

### Code Structure
- **loan_model.py**: Main pipeline with:
  - `DataPreprocessor`: Handles feature engineering (e.g., Income_to_Loan_Ratio), Box-Cox transformation, and categorical encoding.
  - `ModelTrainer`: Trains Logistic Regression and XGBoost with 5-fold CV, hyperparameter tuning, and bias mitigation.
  - `audit_bias`: Computes fairness metrics (DPD, EOD, accuracy, precision, recall, F1).
  - `create_visualizations`: Generates plots for interpretability.
- **ai_risk_report.md**: Summarizes fairness findings and recommendations.

## Methodology
- **Preprocessing**: Box-Cox for skewed features, `OneHotEncoder` for categorical variables, SMOTE for class imbalance.
- **Training**: 5-fold cross-validation, hyperparameter tuning (`GridSearchCV`), and `ExponentiatedGradient` with `DemographicParity`.
- **Fairness**: `fairlearn` metrics (DPD, EOD) for Gender, Race, and Zip_Code_Group, with group filtering (<10 samples excluded).
- **Interpretability**: SHAP, bias-variance plots (inspired by polynomial regression notebook), and fairness visualizations.
- **Production Features**: Logging, timing, and error handling for robustness.

## Results
- **Performance**:
  - Logistic Regression: 0.6284 (5-fold CV accuracy)
  - XGBoost: 0.6200
- **Fairness**:
  - Gender DPD: 0.4167, EOD: 0.4137 (Non-binary recall: 0.3125)
  - Race DPD: 0.2639, EOD: 0.2778 (Native American recall: 0.5000)
  - Zip_Code_Group DPD: 0.1944, EOD: 0.2222
- **Key Findings**:
  - High Gender DPD indicates bias against Non-binary applicants.
  - Low recall for Native American and Non-binary groups suggests underprediction.
  - Redlined Zip_Code_Groups show lower approval rates, reflecting systemic bias.

## Debugging Tips
- Verify dataset paths: Ensure `/kaggle/input/bias-bounty/` contains `loan_access_dataset.csv` and `test.csv`.
- Check `sensitive_features['Gender']` categories: `sensitive_features['Gender'].value_counts()`.
- Inspect encoded features: `X.columns` (e.g., `Gender_Female`, `Race_White`).
- Confirm index alignment: `assert X_train_res.index.equals(sensitive_train_res.index)`.
- Monitor logs for hyperparameter tuning (`GridSearchCV`) and errors.

## Recommendations
- **Bias Mitigation**: Use `EqualizedOdds` constraints to reduce EOD (0.4137).
- **Model Improvement**: Expand hyperparameter tuning and explore ensemble methods.
- **Monitoring**: Implement model drift detection in production.
- **Data**: Oversample minority groups (e.g., Non-binary) before SMOTE.

## Acknowledgments
Inspired by reference notebooks on polynomial regression (bias-variance analysis) and XGBoost with cross-validation (robust preprocessing).
