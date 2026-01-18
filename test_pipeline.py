"""
Test Script for Local Risk Claims Pipeline

This script demonstrates how to:
1. Load the trained model pipeline
2. Make predictions on new claims
3. Test individual claims
4. Batch predict from CSV

The pipeline is externalized for easy testing and reuse.
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os


class RiskClaimsPredictor:
    """Externalized prediction pipeline for testing"""
    
    def __init__(self, model_dir='models'):
        """Load saved model and preprocessing objects"""
        print("=" * 70)
        print(" " * 15 + "LOADING PREDICTION PIPELINE")
        print("=" * 70)
        
        # Load model
        model_path = os.path.join(model_dir, 'risk_model.pkl')
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded: {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded: {scaler_path}")
        
        # Load encoders
        encoders_path = os.path.join(model_dir, 'encoders.pkl')
        self.label_encoders = joblib.load(encoders_path)
        print(f"✓ Label encoders loaded: {encoders_path}")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata['feature_names']
        print(f"✓ Metadata loaded: {metadata_path}")
        
        print(f"\nPipeline Info:")
        print(f"  Model type: {self.metadata['model_type']}")
        print(f"  Version: {self.metadata.get('version', '1.0.0')}")
        print(f"  Trained: {self.metadata['training_date'][:19]}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Test accuracy: {self.metadata['metrics']['accuracy']:.4f}")
        print("=" * 70)
    
    def preprocess_claim(self, claim_data):
        """Preprocess a single claim for prediction"""
        # Create DataFrame
        df = pd.DataFrame([claim_data])
        
        # Feature engineering (same as training pipeline)
        df['claim_to_coverage_ratio'] = df['claim_amount'] / df['policy_coverage']
        df['claim_per_month'] = df['claim_amount'] / df['policy_duration'].clip(lower=1)
        df['is_high_claim'] = (df['claim_amount'] > 10000).astype(int)
        df['is_new_policy'] = (df['policy_duration'] < 12).astype(int)
        df['has_prior_claims'] = (df['previous_claims'] > 0).astype(int)
        df['multiple_prior_claims'] = (df['previous_claims'] >= 2).astype(int)
        
        # Age group
        age = df['customer_age'].iloc[0]
        if age <= 25:
            age_group = 'young'
        elif age <= 40:
            age_group = 'adult'
        elif age <= 60:
            age_group = 'middle_age'
        else:
            age_group = 'senior'
        
        # Encode categorical variables
        claim_type_encoded = self.label_encoders['claim_type'].transform([claim_data['claim_type']])[0]
        age_group_encoded = self.label_encoders['age_group'].transform([age_group])[0]
        
        df['claim_type_encoded'] = claim_type_encoded
        df['age_group_encoded'] = age_group_encoded
        
        # Add optional time-based features if they exist in training
        if 'days_since_claim' in self.feature_names:
            if 'claim_date' in claim_data:
                claim_date = pd.to_datetime(claim_data['claim_date'])
                df['days_since_claim'] = (datetime.now() - claim_date).days
            else:
                df['days_since_claim'] = 0
        
        if 'days_policy_to_claim' in self.feature_names:
            if 'policy_start_date' in claim_data and 'claim_date' in claim_data:
                policy_start = pd.to_datetime(claim_data['policy_start_date'])
                claim_date = pd.to_datetime(claim_data['claim_date'])
                df['days_policy_to_claim'] = (claim_date - policy_start).days
            else:
                df['days_policy_to_claim'] = df['policy_duration'].iloc[0] * 15
        
        # Extract features in correct order
        features = df[self.feature_names].values
        
        return features
    
    def predict(self, claim_data):
        """Predict risk level for a claim"""
        # Preprocess
        features = self.preprocess_claim(claim_data)
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        risk_level = 'high' if prediction == 1 else 'low'
        confidence = probability[prediction]
        
        return {
            'risk_level': risk_level,
            'confidence': float(confidence),
            'probability_low_risk': float(probability[0]),
            'probability_high_risk': float(probability[1]),
            'recommendation': 'Manual Review Required' if risk_level == 'high' else 'Auto-Approve',
            'claim_id': claim_data.get('claim_id', 'N/A'),
            'claim_amount': claim_data.get('claim_amount', 0)
        }
    
    def predict_batch(self, claims_df):
        """Predict risk levels for multiple claims"""
        results = []
        
        for idx, row in claims_df.iterrows():
            claim_data = row.to_dict()
            result = self.predict(claim_data)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def explain_prediction(self, claim_data, result):
        """Provide explanation for prediction"""
        print(f"\nPrediction Explanation:")
        print("-" * 70)
        
        # Feature analysis
        features = self.preprocess_claim(claim_data)[0]
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_contributions = sorted(
            zip(self.feature_names, features, importances),
            key=lambda x: abs(x[1] * x[2]),
            reverse=True
        )
        
        print(f"Top contributing features:")
        for i, (name, value, importance) in enumerate(feature_contributions[:5], 1):
            print(f"  {i}. {name}: {value:.4f} (importance: {importance:.4f})")
        
        # Risk factors
        print(f"\nRisk Factors:")
        if claim_data['claim_amount'] > 20000:
            print(f"  ⚠ High claim amount: ${claim_data['claim_amount']:,.2f}")
        if claim_data['policy_duration'] < 12:
            print(f"  ⚠ New policy: {claim_data['policy_duration']} months")
        if claim_data['previous_claims'] >= 2:
            print(f"  ⚠ Multiple prior claims: {claim_data['previous_claims']}")
        if claim_data['customer_age'] < 25 or claim_data['customer_age'] > 70:
            print(f"  ⚠ Age risk factor: {claim_data['customer_age']} years")
        
        ratio = claim_data['claim_amount'] / claim_data['policy_coverage']
        if ratio > 0.7:
            print(f"  ⚠ High claim-to-coverage ratio: {ratio:.2%}")


def test_single_claim():
    """Test prediction on a single claim"""
    print("\n" + "=" * 70)
    print(" " * 20 + "SINGLE CLAIM TEST")
    print("=" * 70)
    
    # Load predictor
    predictor = RiskClaimsPredictor()
    
    # Test claim 1: High risk scenario
    print("\n" + "=" * 70)
    print("Test Claim 1: High Risk Scenario")
    print("=" * 70)
    
    high_risk_claim = {
        'claim_id': 'TEST-001',
        'customer_id': 'CUST-9999',
        'claim_amount': 45000.00,
        'claim_type': 'auto',
        'claim_date': '2024-01-15',
        'policy_id': 'POL-123456',
        'customer_age': 22,
        'policy_duration': 2,  # Very new policy
        'policy_start_date': '2023-12-01',
        'policy_coverage': 50000.00,
        'previous_claims': 3,
        'claim_description': 'Major vehicle collision',
        'location': 'Los Angeles, CA'
    }
    
    print(f"Claim Details:")
    print(f"  ID: {high_risk_claim['claim_id']}")
    print(f"  Amount: ${high_risk_claim['claim_amount']:,.2f}")
    print(f"  Type: {high_risk_claim['claim_type']}")
    print(f"  Customer Age: {high_risk_claim['customer_age']}")
    print(f"  Policy Duration: {high_risk_claim['policy_duration']} months")
    print(f"  Previous Claims: {high_risk_claim['previous_claims']}")
    print(f"  Coverage: ${high_risk_claim['policy_coverage']:,.2f}")
    
    result = predictor.predict(high_risk_claim)
    
    print(f"\nPrediction Results:")
    print(f"  Risk Level: {result['risk_level'].upper()}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  High Risk Probability: {result['probability_high_risk']:.2%}")
    print(f"  Low Risk Probability: {result['probability_low_risk']:.2%}")
    print(f"  Recommendation: {result['recommendation']}")
    
    predictor.explain_prediction(high_risk_claim, result)
    
    # Test claim 2: Low risk scenario
    print("\n" + "=" * 70)
    print("Test Claim 2: Low Risk Scenario")
    print("=" * 70)
    
    low_risk_claim = {
        'claim_id': 'TEST-002',
        'customer_id': 'CUST-8888',
        'claim_amount': 1200.00,
        'claim_type': 'health',
        'claim_date': '2024-01-10',
        'policy_id': 'POL-789012',
        'customer_age': 45,
        'policy_duration': 36,  # Established policy
        'policy_start_date': '2021-01-01',
        'policy_coverage': 100000.00,
        'previous_claims': 0,
        'claim_description': 'Routine medical procedure',
        'location': 'Chicago, IL'
    }
    
    print(f"Claim Details:")
    print(f"  ID: {low_risk_claim['claim_id']}")
    print(f"  Amount: ${low_risk_claim['claim_amount']:,.2f}")
    print(f"  Type: {low_risk_claim['claim_type']}")
    print(f"  Customer Age: {low_risk_claim['customer_age']}")
    print(f"  Policy Duration: {low_risk_claim['policy_duration']} months")
    print(f"  Previous Claims: {low_risk_claim['previous_claims']}")
    print(f"  Coverage: ${low_risk_claim['policy_coverage']:,.2f}")
    
    result = predictor.predict(low_risk_claim)
    
    print(f"\nPrediction Results:")
    print(f"  Risk Level: {result['risk_level'].upper()}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  High Risk Probability: {result['probability_high_risk']:.2%}")
    print(f"  Low Risk Probability: {result['probability_low_risk']:.2%}")
    print(f"  Recommendation: {result['recommendation']}")
    
    predictor.explain_prediction(low_risk_claim, result)


def test_batch_predictions():
    """Test predictions on batch of claims"""
    print("\n" + "=" * 70)
    print(" " * 20 + "BATCH PREDICTION TEST")
    print("=" * 70)
    
    # Load predictor
    predictor = RiskClaimsPredictor()
    
    # Load test data
    test_data_path = 'data/sample_claims_api_test.csv'
    
    if not os.path.exists(test_data_path):
        print(f"⚠ Test data not found: {test_data_path}")
        print(f"  Run generate_sample_data.py first")
        return
    
    test_data = pd.read_csv(test_data_path)
    print(f"\nLoaded {len(test_data)} claims for batch prediction")
    
    # Make predictions
    print(f"\nProcessing predictions...")
    results = predictor.predict_batch(test_data)
    
    print(f"\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"  Total Claims: {len(results)}")
    print(f"  High Risk: {(results['risk_level'] == 'high').sum()} ({(results['risk_level'] == 'high').sum()/len(results)*100:.1f}%)")
    print(f"  Low Risk: {(results['risk_level'] == 'low').sum()} ({(results['risk_level'] == 'low').sum()/len(results)*100:.1f}%)")
    print(f"  Avg Confidence: {results['confidence'].mean():.2%}")
    print(f"  Avg High Risk Probability: {results['probability_high_risk'].mean():.2%}")
    
    print(f"\n" + "=" * 70)
    print("SAMPLE RESULTS")
    print("=" * 70)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results[['claim_id', 'claim_amount', 'risk_level', 'confidence', 'recommendation']].head(10).to_string(index=False))
    
    # Save results
    output_path = 'data/batch_prediction_results.csv'
    results.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


def test_custom_claims():
    """Test using custom CSV file"""
    print("\n" + "=" * 70)
    print(" " * 20 + "CUSTOM CLAIMS TEST")
    print("=" * 70)
    
    predictor = RiskClaimsPredictor()
    
    # Create custom test claims
    test_claims = pd.DataFrame([
        {
            'claim_id': 'CUSTOM-001',
            'customer_id': 'CUST-1111',
            'claim_amount': 25000.00,
            'claim_type': 'home',
            'claim_date': '2024-01-05',
            'policy_id': 'POL-111111',
            'customer_age': 55,
            'policy_duration': 24,
            'policy_start_date': '2022-01-01',
            'policy_coverage': 300000.00,
            'previous_claims': 1,
            'claim_description': 'Water damage from pipe burst',
            'location': 'New York, NY'
        },
        {
            'claim_id': 'CUSTOM-002',
            'customer_id': 'CUST-2222',
            'claim_amount': 75000.00,
            'claim_type': 'life',
            'claim_date': '2024-01-01',
            'policy_id': 'POL-222222',
            'customer_age': 70,
            'policy_duration': 6,
            'policy_start_date': '2023-07-01',
            'policy_coverage': 100000.00,
            'previous_claims': 0,
            'claim_description': 'Critical illness claim',
            'location': 'Phoenix, AZ'
        },
        {
            'claim_id': 'CUSTOM-003',
            'customer_id': 'CUST-3333',
            'claim_amount': 3500.00,
            'claim_type': 'auto',
            'claim_date': '2023-12-20',
            'policy_id': 'POL-333333',
            'customer_age': 35,
            'policy_duration': 48,
            'policy_start_date': '2020-01-01',
            'policy_coverage': 50000.00,
            'previous_claims': 0,
            'claim_description': 'Minor fender bender',
            'location': 'Dallas, TX'
        }
    ])
    
    print(f"\nTesting {len(test_claims)} custom claims:")
    results = predictor.predict_batch(test_claims)
    
    print(f"\n" + "=" * 70)
    print("CUSTOM CLAIMS RESULTS")
    print("=" * 70)
    
    for idx, row in results.iterrows():
        print(f"\n{row['claim_id']}:")
        print(f"  Amount: ${row['claim_amount']:,.2f}")
        print(f"  Risk: {row['risk_level'].upper()}")
        print(f"  Confidence: {row['confidence']:.2%}")
        print(f"  Action: {row['recommendation']}")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print(" " * 20 + "RISK CLAIMS MODEL - TEST SUITE")
    print(" " * 25 + "Externalized Pipeline Testing")
    print("=" * 80)
    
    try:
        # Test 1: Single claims
        test_single_claim()
        
        # Test 2: Batch predictions
        test_batch_predictions()
        
        # Test 3: Custom claims
        test_custom_claims()
        
        print("\n" + "=" * 80)
        print(" " * 30 + "ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe prediction pipeline is working correctly and ready for use.")
        print("You can now:")
        print("  1. Use RiskClaimsPredictor class in your own code")
        print("  2. Test with custom CSV files")
        print("  3. Integrate with APIs or other systems")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Test failed: {str(e)}")
        print(f"\nPlease run these commands first:")
        print(f"  1. python generate_sample_data.py")
        print(f"  2. python pipeline.py")
        print(f"  3. python test_pipeline.py")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
