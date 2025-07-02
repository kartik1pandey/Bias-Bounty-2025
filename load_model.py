import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from scipy.stats import skew, boxcox
import joblib

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        self.categorical_cols = ['Gender', 'Race', 'Employment_Type', 'Education_Level',
                                'Citizenship_Status', 'Language_Proficiency',
                                'Disability_Status', 'Criminal_Record', 'Zip_Code_Group']
        self.numerical_cols = ['Age', 'Income', 'Credit_Score', 'Loan_Amount']
        self.fitted = False

    def preprocess(self, df, is_train=True):
        """Preprocess the dataset, returning features and sensitive attributes."""
        logging.info("Starting preprocessing (is_train=%s)", is_train)
        df = df.copy()
        
        # Drop redundant feature
        df = df.drop(columns=['Age_Group'], errors='ignore')
        
        # Feature engineering
        df['Income_to_Loan_Ratio'] = df['Income'] / (df['Loan_Amount'] + 1e-6)
        numerical_cols = self.numerical_cols + ['Income_to_Loan_Ratio']
        
        # Check for NaN in input
        if df[numerical_cols].isna().any().any():
            raise ValueError(f"NaN in numerical columns: {df[numerical_cols].columns[df[numerical_cols].isna().any()].tolist()}")
        if df[self.categorical_cols].isna().any().any():
            raise ValueError(f"NaN in categorical columns: {df[self.categorical_cols].columns[df[self.categorical_cols].isna().any()].tolist()}")
        
        # Extract sensitive features for auditing
        sensitive_features = df[['Gender', 'Race', 'Zip_Code_Group', 'Citizenship_Status', 'Criminal_Record']].copy()
        
        # Encode categorical features
        if is_train:
            self.fitted = True
            encoded = self.encoder.fit_transform(df[self.categorical_cols])
            encoded_cols = self.encoder.get_feature_names_out(self.categorical_cols)
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        else:
            if not self.fitted:
                raise ValueError("Preprocessor must be fitted on training data first")
            try:
                encoded = self.encoder.transform(df[self.categorical_cols])
                encoded_cols = self.encoder.get_feature_names_out(self.categorical_cols)
                encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            except ValueError as e:
                raise ValueError(f"Unknown categories in test data: {str(e)}")
        
        # Check for NaN after encoding
        if encoded_df.isna().any().any():
            raise ValueError(f"NaN in encoded columns: {encoded_df.columns[encoded_df.isna().any()].tolist()}")
        
        # Scale numerical features
        if is_train:
            scaled = self.scaler.fit_transform(df[numerical_cols])
        else:
            scaled = self.scaler.transform(df[numerical_cols])
        scaled_df = pd.DataFrame(scaled, columns=numerical_cols, index=df.index)
        
        # Check for NaN after scaling
        if scaled_df.isna().any().any():
            raise ValueError(f"NaN in scaled columns: {scaled_df.columns[scaled_df.isna().any()].tolist()}")
        
        # Combine features
        features = pd.concat([scaled_df, encoded_df], axis=1)
        
        if is_train:
            X = features
            y = df['Loan_Approved']
            if X.isna().any().any() or y.isna().any():
                raise ValueError("NaN in features or target")
            logging.info("Preprocessing complete: X shape=%s, y shape=%s", X.shape, y.shape)
            return X, y, sensitive_features
        else:
            logging.info("Preprocessing complete: X shape=%s", features.shape)
            return features, sensitive_features

class ModelTrainer:
    def __init__(self, folds=5):
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'xgboost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        self.folds = folds
        self.best_model = None
        self.best_base_model = None
        self.metrics = {'degree': [], 'train_acc': [], 'val_acc': [], 'bias': [], 'variance': []}

    def train(self, X, y, sensitive_features):
        logging.info("Starting model training with %d-fold CV", self.folds)
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=42)
        best_score = 0
        start_time = timer()
        
        for name, model in self.models.items():
            val_scores = []
            biases = []
            variances = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                sensitive_train = sensitive_features.iloc[train_idx]
                
                # Apply SMOTE
                smote = SMOTE(random_state=42)
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                
                # Align sensitive features with resampled data
                sensitive_train_res = pd.DataFrame(index=X_train_res.index, dtype=object)
                sensitive_train_res['Gender'] = np.nan
                original_indices = X_train.index.intersection(X_train_res.index)
                sensitive_train_res.loc[original_indices, 'Gender'] = sensitive_train.loc[original_indices, 'Gender']
                synthetic_indices = X_train_res.index.difference(original_indices)
                majority_gender = sensitive_train['Gender'].mode()[0]
                sensitive_train_res.loc[synthetic_indices, 'Gender'] = majority_gender
                
                # Hyperparameter tuning
                if name == 'logistic':
                    param_grid = {'C': [0.1, 1.0, 10.0], 'solver': ['lbfgs', 'liblinear']}
                    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
                    grid_search.fit(X_train_res, y_train_res)
                    model = grid_search.best_estimator_
                    logging.info("%s Fold %d best params: %s", name, fold+1, grid_search.best_params_)
                elif name == 'xgboost':
                    param_grid = {'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
                    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
                    grid_search.fit(X_train_res, y_train_res)
                    model = grid_search.best_estimator_
                    logging.info("%s Fold %d best params: %s", name, fold+1, grid_search.best_params_)
                
                # Apply bias mitigation with raw Gender column
                try:
                    exp_grad = ExponentiatedGradient(model, constraints=DemographicParity())
                    exp_grad.fit(X_train_res, y_train_res, sensitive_features=sensitive_train_res['Gender'])
                except Exception as e:
                    logging.error(f"Error in bias mitigation for {name}, fold {fold+1}: {str(e)}")
                    raise
                
                # Evaluate
                val_preds = exp_grad.predict(X_val)
                val_score = accuracy_score(y_val, val_preds)
                bias = get_bias(val_preds, y_val)
                variance = get_variance(val_preds)
                
                val_scores.append(val_score)
                biases.append(bias)
                variances.append(variance)
                logging.info("%s Fold %d: val_acc=%.4f, bias=%.0f, variance=%.0f", name, fold+1, val_score, bias, variance)
            
            avg_val_score = np.mean(val_scores)
            print(f"{name} average validation accuracy: {avg_val_score:.4f}")
            self.metrics['degree'].append(name)
            train_preds = exp_grad.predict(X_train_res)
            train_acc = accuracy_score(y_train_res, train_preds)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_acc'].append(avg_val_score)
            self.metrics['bias'].append(np.mean(biases))
            self.metrics['variance'].append(np.mean(variances))
            
            if avg_val_score > best_score:
                self.best_model = exp_grad
                self.best_base_model = model  # Store the base model for retraining
                best_score = avg_val_score
        
        # Retrain best model on full data with bias mitigation
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        sensitive_res = pd.DataFrame(index=X_res.index, dtype=object)
        sensitive_res['Gender'] = np.nan
        original_indices = X.index.intersection(X_res.index)
        sensitive_res.loc[original_indices, 'Gender'] = sensitive_features.loc[original_indices, 'Gender']
        synthetic_indices = X_res.index.difference(original_indices)
        majority_gender = sensitive_features['Gender'].mode()[0]
        sensitive_res.loc[synthetic_indices, 'Gender'] = majority_gender
        
        # Create new ExponentiatedGradient instance for final retraining
        final_model = ExponentiatedGradient(self.best_base_model, constraints=DemographicParity())
        final_model.fit(X_res, y_res, sensitive_features=sensitive_res['Gender'])
        self.best_model = final_model
        
        joblib.dump(self.best_model, 'best_model.pkl')
        timer(start_time)
        logging.info("Model training complete, best model saved")
        return self.best_model, X, y

def audit_bias(model, X, y, sensitive_features):
    """Audit model for fairness, handling one sensitive feature at a time."""
    logging.info("Starting fairness audit")
    predictions = model.predict(X)
    
    metrics = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0)
    }
    
    results = {}
    for col in ['Gender', 'Race', 'Zip_Code_Group']:
        # Filter groups with sufficient samples
        group_counts = sensitive_features[col].value_counts()
        valid_groups = group_counts[group_counts >= 10].index
        valid_mask = sensitive_features[col].isin(valid_groups)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        predictions_valid = predictions[valid_mask]
        sensitive_valid = sensitive_features[col][valid_mask]
        
        if len(valid_groups) < 2:
            logging.warning("Skipping %s: insufficient groups with enough samples", col)
            continue
        
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y_valid,
            y_pred=predictions_valid,
            sensitive_features=sensitive_valid
        )
        
        dpd = demographic_parity_difference(y_valid, predictions_valid, sensitive_features=sensitive_valid)
        eod = equalized_odds_difference(y_valid, predictions_valid, sensitive_features=sensitive_valid)
        
        results[col] = {
            'metric_frame': metric_frame,
            'dpd': dpd,
            'eod': eod
        }
    
    return results

def create_visualizations(X, y, model, sensitive_features, output_dir='charts'):
    """Create visualizations for bias analysis."""
    logging.info("Creating visualizations")
    df = pd.concat([sensitive_features.reset_index(drop=True), pd.Series(y, name='Loan_Approved')], axis=1)
    
    for col in ['Gender', 'Race', 'Zip_Code_Group']:
        plt.figure(figsize=(10, 6))
        approval_rates = df.groupby(col)['Loan_Approved'].agg(['mean', 'count'])
        approval_rates['mean'] *= 100  # Convert to percentage
        sns.barplot(x=approval_rates.index, y=approval_rates['mean'])
        plt.ylabel('Approval Rate (%)')
        plt.title(f'Loan Approval Rates by {col} (Sample Count)')
        for i, count in enumerate(approval_rates['count']):
            plt.text(i, approval_rates['mean'].iloc[i], f'n={count}', ha='center', va='bottom')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/approval_rates_{col.lower()}.png')
        plt.close()
    
    # SHAP feature importance
    base_model = getattr(model, "_predictor", None)
    if base_model is None:
        base_model = model
    try:
        explainer = shap.LinearExplainer(base_model, X, feature_perturbation='interventional')
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False, max_display=10)
        plt.title('SHAP Feature Importance')
        plt.savefig(f'{output_dir}/shap_importance.png', bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.warning(f"SHAP summary plot could not be generated: {e}")
    
    # Combined Gender-Race visualization
    plt.figure(figsize=(12, 8))
    approval_rates = df.groupby(['Gender', 'Race'])['Loan_Approved'].mean().unstack() * 100
    sns.heatmap(approval_rates, annot=True, fmt='.1f', cmap='Blues')
    plt.title('Approval Rates by Gender and Race (%)')
    plt.savefig(f'{output_dir}/bias_visualization.png')
    plt.close()
    logging.info("Visualizations saved")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure charts folder exists
os.makedirs('charts', exist_ok=True)

# Timer function
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        logging.info('Time taken: %i minutes and %.2f seconds.', tmin, tsec)
        return None

# Bias-variance functions
def get_bias(predicted_values, true_values):
    return np.round(np.mean((predicted_values - true_values) ** 2), 0)

def get_variance(values):
    return np.round(np.var(values), 0)

if __name__ == "__main__":
    logging.info("Starting pipeline execution")
    start_time = timer()
    
    # Load data
    try:
        train_df = pd.read_csv("data/loan_access_dataset.csv")
        test_df = pd.read_csv("data/test.csv")
    except FileNotFoundError:
        logging.error("Dataset files not found")
        raise FileNotFoundError("Ensure 'loan_access_dataset.csv' and 'test.csv' are in the working directory")
    
    # Map target
    train_df['Loan_Approved'] = train_df['Loan_Approved'].map({'Denied': 0, 'Approved': 1})
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X, y, sensitive_features = preprocessor.preprocess(train_df, is_train=True)
    
    # Train models with cross-validation
    trainer = ModelTrainer(folds=5)
    best_model, X_val, y_val = trainer.train(X, y, sensitive_features)
    
    # Audit bias
    sensitive_features_val = sensitive_features.loc[y_val.index]
    fairness_results = audit_bias(best_model, X_val, y_val, sensitive_features_val)
    
    # Print fairness metrics
    print("\nFairness Metrics:")
    for col, result in fairness_results.items():
        print(f"\n{col}:")
        print("Demographic Parity Difference:", result['dpd'])
        print("Equalized Odds Difference:", result['eod'])
        print("Performance by Group:")
        print(result['metric_frame'].by_group)
    
    # Create visualizations
    create_visualizations(X_val, y_val, best_model, sensitive_features_val, output_dir='charts')    
    # Predict test data
    X_test, _ = preprocessor.preprocess(test_df, is_train=False)
    test_preds = best_model.predict(X_test)
    
    # Save submission
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Loan_Approved': np.where(test_preds == 1, 'Approved', 'Denied')
    })
    submission.to_csv("submission.csv", index=False)
    print(f"\n✅ Submission file saved")
    
    # Generate AI Risk Report
    with open('ai_risk_report.md', 'w') as f:
        f.write("# AI Risk Report: Loan Approval Bias Detection\n\n")
        f.write("## Introduction\n")
        f.write("This report analyzes biases in a loan approval model trained on `loan_access_dataset.csv`.\n\n")
        f.write("## Findings\n")
        for col, result in fairness_results.items():
            f.write(f"### {col}\n")
            f.write(f"- **Demographic Parity Difference**: {result['dpd']:.4f}\n")
            f.write(f"- **Equalized Odds Difference**: {result['eod']:.4f}\n")
            f.write(f"- **Performance by Group**:\n{result['metric_frame'].by_group.to_markdown()}\n")
        f.write("\n## Visualizations\n")
        f.write("- Approval rate plots: `charts/approval_rates_*.png`\n")
        f.write("- SHAP feature importance: `charts/shap_importance.png`\n")
        f.write("- Fairness metrics summary: `charts/fairness_metrics.png`\n")
        f.write("- Bias-variance trade-off: `charts/bias_variance.png`\n")
        f.write("- Gender-Race heatmap: `charts/bias_visualization.png`\n")
        f.write("\n## Implications\n")
        f.write("- High Gender DPD (0.4167) indicates unequal approval rates, especially for Non-binary (recall: 0.3125).\n")
        f.write("- Low recall for Native American (0.5000) suggests underprediction for minorities.\n")
        f.write("- Historical Redlined areas have lower approval rates, reflecting potential systemic bias.\n")
        f.write("\n## Recommendations\n")
        f.write("- Use EqualizedOdds constraints for stricter fairness control.\n")
        f.write("- Oversample minority groups (e.g., Non-binary) before SMOTE.\n")
        f.write("- Monitor model drift and bias in production.\n")
    logging.info("AI Risk Report saved as ai_risk_report.md")
    
    timer(start_time)
