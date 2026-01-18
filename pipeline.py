"""
Local ML Pipeline for Risk Claims Classification

A lightweight implementation of all 10 ML pipeline stages that runs entirely on local system
with minimal dependencies and compute requirements.

This pipeline demonstrates:
- Stage 1: Data Ingestion
- Stage 2: Data Validation
- Stage 3: Data Preprocessing
- Stage 4: Feature Engineering
- Stage 5: Model Training
- Stage 6: Model Evaluation
- Stage 7: Model Selection
- Stage 8: Model Deployment
- Stage 9: Monitoring
- Stage 10: Model Retraining
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import joblib
import json
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


class LocalRiskClaimsPipeline:
    """Complete ML pipeline for local deployment"""
    
    def __init__(self, config_path='config/pipeline_config.json'):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_names = None
        self.metrics = {}
        
    def _load_config(self, config_path):
        """Load configuration file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'data_path': 'data/sample_claims_train.csv',
            'test_size': 0.2,
            'random_state': 42,
            'model_params': {
                'n_estimators': 50,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            },
            'model_path': 'models/risk_model.pkl',
            'scaler_path': 'models/scaler.pkl',
            'encoders_path': 'models/encoders.pkl',
            'metadata_path': 'models/metadata.json'
        }
    
    # ============= STAGE 1: DATA INGESTION =============
    def stage1_ingest_data(self, data_path=None):
        """
        Stage 1: Data Ingestion
        Load data from local CSV file
        """
        print("\n" + "=" * 70)
        print("STAGE 1: DATA INGESTION")
        print("=" * 70)
        
        path = data_path or self.config['data_path']
        print(f"Loading data from: {path}")
        
        self.df = pd.read_csv(path)
        print(f"✓ Loaded {len(self.df)} records")
        print(f"✓ Columns ({len(self.df.columns)}): {', '.join(self.df.columns[:5])}...")
        print(f"✓ Shape: {self.df.shape}")
        print(f"✓ Memory usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        return self.df
    
    # ============= STAGE 2: DATA VALIDATION =============
    def stage2_validate_data(self):
        """
        Stage 2: Data Validation
        Check data quality and schema
        """
        print("\n" + "=" * 70)
        print("STAGE 2: DATA VALIDATION")
        print("=" * 70)
        
        required_columns = [
            'claim_amount', 'claim_type', 'customer_age', 'policy_duration',
            'policy_coverage', 'previous_claims', 'risk_level'
        ]
        
        # Schema validation
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        print(f"✓ Schema validation passed - all {len(required_columns)} required columns present")
        
        # Data quality checks
        null_counts = self.df[required_columns].isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            print(f"⚠ Warning: Found {total_nulls} null values")
            for col, count in null_counts[null_counts > 0].items():
                print(f"    {col}: {count}")
        else:
            print(f"✓ No null values found")
        
        # Value range checks
        age_issues = self.df[(self.df['customer_age'] < 18) | (self.df['customer_age'] > 100)]
        if len(age_issues) > 0:
            print(f"⚠ Warning: {len(age_issues)} records with invalid age")
        else:
            print(f"✓ Age validation passed (range: 18-100)")
        
        amount_issues = self.df[self.df['claim_amount'] <= 0]
        if len(amount_issues) > 0:
            print(f"⚠ Warning: {len(amount_issues)} records with invalid claim amount")
        else:
            print(f"✓ Claim amount validation passed (all > 0)")
        
        # Check target variable distribution
        risk_dist = self.df['risk_level'].value_counts()
        print(f"✓ Target distribution:")
        for risk, count in risk_dist.items():
            print(f"    {risk}: {count} ({count/len(self.df)*100:.1f}%)")
        
        print(f"✓ Data validation complete")
        
        return True
    
    # ============= STAGE 3: DATA PREPROCESSING =============
    def stage3_preprocess_data(self):
        """
        Stage 3: Data Preprocessing
        Clean and transform data
        """
        print("\n" + "=" * 70)
        print("STAGE 3: DATA PREPROCESSING")
        print("=" * 70)
        
        initial_count = len(self.df)
        
        # Handle missing values
        numeric_cols = ['claim_amount', 'customer_age', 'policy_duration', 
                       'policy_coverage', 'previous_claims']
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"✓ Filled {col} nulls with median: {median_val}")
        
        # Convert dates to numeric features
        if 'claim_date' in self.df.columns:
            self.df['claim_date'] = pd.to_datetime(self.df['claim_date'])
            self.df['days_since_claim'] = (datetime.now() - self.df['claim_date']).dt.days
            print(f"✓ Converted claim_date to days_since_claim")
        
        if 'policy_start_date' in self.df.columns:
            self.df['policy_start_date'] = pd.to_datetime(self.df['policy_start_date'])
            if 'claim_date' in self.df.columns:
                self.df['days_policy_to_claim'] = (
                    self.df['claim_date'] - self.df['policy_start_date']
                ).dt.days
                print(f"✓ Calculated days_policy_to_claim")
        
        # Remove extreme outliers (optional - keeping data for small dataset)
        # Using IQR method for claim amounts
        Q1 = self.df['claim_amount'].quantile(0.25)
        Q3 = self.df['claim_amount'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 3 * IQR  # Only remove extreme outliers
        outliers = self.df[self.df['claim_amount'] > outlier_threshold]
        
        print(f"✓ Outlier analysis: {len(outliers)} extreme outliers detected (keeping all data)")
        print(f"✓ Preprocessing complete - {len(self.df)} of {initial_count} records retained")
        
        return self.df
    
    # ============= STAGE 4: FEATURE ENGINEERING =============
    def stage4_engineer_features(self):
        """
        Stage 4: Feature Engineering
        Create derived features for better predictions
        """
        print("\n" + "=" * 70)
        print("STAGE 4: FEATURE ENGINEERING")
        print("=" * 70)
        
        # Numeric features
        self.df['claim_to_coverage_ratio'] = (
            self.df['claim_amount'] / self.df['policy_coverage']
        )
        
        self.df['claim_per_month'] = (
            self.df['claim_amount'] / self.df['policy_duration'].clip(lower=1)
        )
        
        self.df['is_high_claim'] = (self.df['claim_amount'] > 
                                    self.df['claim_amount'].quantile(0.75)).astype(int)
        
        self.df['is_new_policy'] = (self.df['policy_duration'] < 12).astype(int)
        
        # Age grouping
        self.df['age_group'] = pd.cut(
            self.df['customer_age'], 
            bins=[0, 25, 40, 60, 100], 
            labels=['young', 'adult', 'middle_age', 'senior']
        )
        
        # Risk indicators
        self.df['has_prior_claims'] = (self.df['previous_claims'] > 0).astype(int)
        self.df['multiple_prior_claims'] = (self.df['previous_claims'] >= 2).astype(int)
        
        print(f"✓ Created 7 derived features")
        
        # Categorical encoding
        categorical_cols = ['claim_type', 'age_group']
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"✓ Encoded {col}: {len(le.classes_)} categories")
        
        # Define feature set
        self.feature_names = [
            # Original features
            'claim_amount', 'customer_age', 'policy_duration', 'policy_coverage',
            'previous_claims',
            # Derived features
            'claim_to_coverage_ratio', 'claim_per_month', 'is_high_claim', 
            'is_new_policy', 'has_prior_claims', 'multiple_prior_claims',
            # Encoded features
            'claim_type_encoded', 'age_group_encoded'
        ]
        
        # Add optional time-based features if they exist
        if 'days_since_claim' in self.df.columns:
            self.feature_names.append('days_since_claim')
        if 'days_policy_to_claim' in self.df.columns:
            self.feature_names.append('days_policy_to_claim')
        
        print(f"✓ Feature engineering complete - {len(self.feature_names)} total features")
        print(f"  Features: {', '.join(self.feature_names[:5])}...")
        
        return self.df
    
    # ============= STAGE 5: MODEL TRAINING =============
    def stage5_train_model(self):
        """
        Stage 5: Model Training
        Train Random Forest classifier
        """
        print("\n" + "=" * 70)
        print("STAGE 5: MODEL TRAINING")
        print("=" * 70)
        
        # Prepare features and target
        X = self.df[self.feature_names]
        y = (self.df['risk_level'] == 'high').astype(int)  # 1 for high risk, 0 for low risk
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: High={y.sum()}, Low={(~y.astype(bool)).sum()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        print(f"\n✓ Data split complete:")
        print(f"  Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"  High risk (train): {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
        print(f"  Low risk (train): {(~y_train.astype(bool)).sum()} ({(~y_train.astype(bool)).sum()/len(y_train)*100:.1f}%)")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n✓ Features scaled using StandardScaler")
        
        # Train model
        print(f"\nTraining Random Forest Classifier...")
        self.model = RandomForestClassifier(**self.config['model_params'])
        self.model.fit(X_train_scaled, y_train)
        
        print(f"✓ Model trained successfully!")
        print(f"  Algorithm: {self.model.__class__.__name__}")
        print(f"  Estimators: {self.config['model_params']['n_estimators']}")
        print(f"  Max depth: {self.config['model_params']['max_depth']}")
        print(f"  Training samples: {len(X_train)}")
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.X_train = X_train_scaled
        self.y_train = y_train
        
        return self.model
    
    # ============= STAGE 6: MODEL EVALUATION =============
    def stage6_evaluate_model(self):
        """
        Stage 6: Model Evaluation
        Calculate comprehensive performance metrics
        """
        print("\n" + "=" * 70)
        print("STAGE 6: MODEL EVALUATION")
        print("=" * 70)
        
        # Predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        y_pred_proba_test = self.model.predict_proba(self.X_test)
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(self.y_train, y_pred_train),
            'precision': precision_score(self.y_train, y_pred_train, zero_division=0),
            'recall': recall_score(self.y_train, y_pred_train, zero_division=0),
            'f1': f1_score(self.y_train, y_pred_train, zero_division=0)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred_test),
            'precision': precision_score(self.y_test, y_pred_test, zero_division=0),
            'recall': recall_score(self.y_test, y_pred_test, zero_division=0),
            'f1': f1_score(self.y_test, y_pred_test, zero_division=0)
        }
        
        print("Training Metrics:")
        print("-" * 70)
        for metric, value in train_metrics.items():
            print(f"  {metric.capitalize():12s}: {value:.4f}")
        
        print("\nTest Metrics:")
        print("-" * 70)
        for metric, value in test_metrics.items():
            print(f"  {metric.capitalize():12s}: {value:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        print(f"\nConfusion Matrix:")
        print("-" * 70)
        print(f"                 Predicted Low  Predicted High")
        print(f"  Actual Low          {cm[0,0]:4d}           {cm[0,1]:4d}")
        print(f"  Actual High         {cm[1,0]:4d}           {cm[1,1]:4d}")
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"\nAdditional Metrics:")
        print("-" * 70)
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        print(f"  Specificity:     {specificity:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Important Features:")
        print("-" * 70)
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        # Store metrics
        self.metrics = {
            'train': train_metrics,
            'test': test_metrics,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance.to_dict('records'),
            'classification_report': classification_report(self.y_test, y_pred_test, 
                                                          target_names=['Low Risk', 'High Risk'],
                                                          output_dict=True)
        }
        
        print(f"\n✓ Evaluation complete")
        
        return self.metrics
    
    # ============= STAGE 7: MODEL SELECTION =============
    def stage7_select_model(self):
        """
        Stage 7: Model Selection
        Select best performing model (in this version, we have one model)
        """
        print("\n" + "=" * 70)
        print("STAGE 7: MODEL SELECTION")
        print("=" * 70)
        
        print(f"Model Comparison:")
        print("-" * 70)
        print(f"  Candidate: Random Forest Classifier")
        print(f"  Test Accuracy: {self.metrics['test']['accuracy']:.4f}")
        print(f"  Test Precision: {self.metrics['test']['precision']:.4f}")
        print(f"  Test Recall: {self.metrics['test']['recall']:.4f}")
        print(f"  Test F1 Score: {self.metrics['test']['f1']:.4f}")
        
        print(f"\n✓ Selected model: Random Forest Classifier")
        print(f"  Reason: Best overall performance")
        print(f"  Final Test Accuracy: {self.metrics['test']['accuracy']:.4f}")
        print(f"  Final Test F1 Score: {self.metrics['test']['f1']:.4f}")
        
        return self.model
    
    # ============= STAGE 8: MODEL DEPLOYMENT =============
    def stage8_deploy_model(self):
        """
        Stage 8: Model Deployment
        Save model and preprocessing objects for serving
        """
        print("\n" + "=" * 70)
        print("STAGE 8: MODEL DEPLOYMENT")
        print("=" * 70)
        
        os.makedirs('models', exist_ok=True)
        
        # Save model
        joblib.dump(self.model, self.config['model_path'])
        model_size = os.path.getsize(self.config['model_path']) / 1024
        print(f"✓ Model saved: {self.config['model_path']} ({model_size:.2f} KB)")
        
        # Save scaler
        joblib.dump(self.scaler, self.config['scaler_path'])
        scaler_size = os.path.getsize(self.config['scaler_path']) / 1024
        print(f"✓ Scaler saved: {self.config['scaler_path']} ({scaler_size:.2f} KB)")
        
        # Save encoders
        joblib.dump(self.label_encoders, self.config['encoders_path'])
        encoder_size = os.path.getsize(self.config['encoders_path']) / 1024
        print(f"✓ Encoders saved: {self.config['encoders_path']} ({encoder_size:.2f} KB)")
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'model_type': self.model.__class__.__name__,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(self.y_train),
            'test_samples': len(self.y_test),
            'metrics': self.metrics['test'],
            'config': self.config,
            'version': '1.0.0'
        }
        
        with open(self.config['metadata_path'], 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved: {self.config['metadata_path']}")
        
        total_size = model_size + scaler_size + encoder_size
        print(f"\n✓ Model deployment complete")
        print(f"  Total deployment size: {total_size:.2f} KB")
        
        return True
    
    # ============= STAGE 9: MONITORING =============
    def stage9_monitor_model(self, new_data_path=None):
        """
        Stage 9: Monitoring
        Basic monitoring - check model performance and data drift
        """
        print("\n" + "=" * 70)
        print("STAGE 9: MONITORING")
        print("=" * 70)
        
        if new_data_path and os.path.exists(new_data_path):
            new_df = pd.read_csv(new_data_path)
            print(f"✓ Loaded monitoring data: {len(new_df)} records")
            
            # Check for data drift (simple version)
            print("\nData Distribution Comparison:")
            print("-" * 70)
            
            # Compare claim amounts
            train_mean = self.df['claim_amount'].mean()
            new_mean = new_df['claim_amount'].mean()
            drift_pct = abs(new_mean - train_mean) / train_mean * 100
            
            print(f"  Claim Amount:")
            print(f"    Training mean: ${train_mean:,.2f}")
            print(f"    New data mean: ${new_mean:,.2f}")
            print(f"    Drift: {drift_pct:.1f}%", end="")
            
            if drift_pct > 20:
                print(" ⚠ WARNING: Significant drift detected!")
            else:
                print(" ✓")
            
            # Compare age distribution
            train_age_mean = self.df['customer_age'].mean()
            new_age_mean = new_df['customer_age'].mean()
            age_drift_pct = abs(new_age_mean - train_age_mean) / train_age_mean * 100
            
            print(f"\n  Customer Age:")
            print(f"    Training mean: {train_age_mean:.1f}")
            print(f"    New data mean: {new_age_mean:.1f}")
            print(f"    Drift: {age_drift_pct:.1f}%", end="")
            
            if age_drift_pct > 15:
                print(" ⚠ WARNING: Significant drift detected!")
            else:
                print(" ✓")
            
            # Recommendations
            print(f"\nMonitoring Recommendations:")
            print("-" * 70)
            if drift_pct > 20 or age_drift_pct > 15:
                print("  ⚠ Consider retraining the model with new data")
            else:
                print("  ✓ No immediate action required")
        else:
            print(f"✓ Monitoring configuration ready")
            print(f"  Provide new data path to monitor for drift")
            print(f"\nMonitoring Metrics:")
            print("-" * 70)
            print(f"  Model accuracy: {self.metrics['test']['accuracy']:.4f}")
            print(f"  Training date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Samples monitored: {len(self.df)}")
        
        print(f"\n✓ Monitoring stage complete")
        
        return True
    
    # ============= STAGE 10: MODEL RETRAINING =============
    def stage10_retrain_model(self, new_data_path):
        """
        Stage 10: Model Retraining
        Retrain model with new data when drift detected
        """
        print("\n" + "=" * 70)
        print("STAGE 10: MODEL RETRAINING")
        print("=" * 70)
        
        if not os.path.exists(new_data_path):
            print(f"✓ Retraining ready (no new data provided)")
            print(f"  Trigger: Significant data drift or performance degradation")
            print(f"  Action: Combine old and new data, retrain all stages")
            return self.model
        
        print(f"✓ Retraining trigger: New data available")
        print(f"  Loading new data from: {new_data_path}")
        
        # Load and combine data
        new_df = pd.read_csv(new_data_path)
        old_size = len(self.df)
        new_size = len(new_df)
        
        combined_df = pd.concat([self.df, new_df], ignore_index=True)
        
        print(f"  Old data: {old_size} records")
        print(f"  New data: {new_size} records")
        print(f"  Combined: {len(combined_df)} records ({(new_size/old_size)*100:.1f}% increase)")
        
        # Re-run pipeline
        print(f"\nRe-running pipeline stages...")
        self.df = combined_df
        
        self.stage2_validate_data()
        self.stage3_preprocess_data()
        self.stage4_engineer_features()
        self.stage5_train_model()
        self.stage6_evaluate_model()
        self.stage7_select_model()
        self.stage8_deploy_model()
        
        print(f"\n✓ Model retraining complete")
        print(f"  New model accuracy: {self.metrics['test']['accuracy']:.4f}")
        
        return self.model
    
    def run_full_pipeline(self, retrain_data_path=None, monitor_data_path=None):
        """Run all 10 stages of the ML pipeline"""
        print("\n" + "=" * 80)
        print(" " * 15 + "LOCAL RISK CLAIMS ML PIPELINE")
        print(" " * 20 + "All 10 Stages - Local Deployment")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python version: {__import__('sys').version.split()[0]}")
        print("=" * 80)
        
        try:
            # Run all stages
            self.stage1_ingest_data()
            self.stage2_validate_data()
            self.stage3_preprocess_data()
            self.stage4_engineer_features()
            self.stage5_train_model()
            self.stage6_evaluate_model()
            self.stage7_select_model()
            self.stage8_deploy_model()
            self.stage9_monitor_model(monitor_data_path)
            
            if retrain_data_path:
                self.stage10_retrain_model(retrain_data_path)
            else:
                # Just show retraining is ready
                print("\n" + "=" * 70)
                print("STAGE 10: MODEL RETRAINING")
                print("=" * 70)
                print("✓ Retraining capability ready")
                print("  Provide new data path to trigger retraining")
            
            # Final summary
            print("\n" + "=" * 80)
            print(" " * 25 + "PIPELINE COMPLETE!")
            print("=" * 80)
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nFinal Model Performance:")
            print(f"  Test Accuracy:  {self.metrics['test']['accuracy']:.4f}")
            print(f"  Test Precision: {self.metrics['test']['precision']:.4f}")
            print(f"  Test Recall:    {self.metrics['test']['recall']:.4f}")
            print(f"  Test F1 Score:  {self.metrics['test']['f1']:.4f}")
            print(f"\nModel saved to: {self.config['model_path']}")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution"""
    pipeline = LocalRiskClaimsPipeline()
    pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()
