"""
Generate Sample Claims Data for Local Testing

This script creates a realistic but small dataset for testing the ML pipeline locally.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_sample_claims(n_samples=1000):
    """Generate sample insurance claims data"""
    
    # Claim types and their base risk probabilities
    claim_types = {
        'auto': 0.3,
        'home': 0.2,
        'health': 0.4,
        'life': 0.1
    }
    
    # Locations with different risk profiles
    locations = ['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 
                 'Phoenix, AZ', 'Philadelphia, PA', 'San Antonio, TX', 'Dallas, TX']
    
    data = []
    
    for i in range(n_samples):
        claim_type = random.choices(
            list(claim_types.keys()), 
            weights=list(claim_types.values())
        )[0]
        
        # Generate customer data
        customer_age = int(np.random.normal(45, 15))
        customer_age = max(18, min(85, customer_age))
        
        # Policy duration in months
        policy_duration = int(np.random.exponential(24))
        policy_duration = max(1, min(120, policy_duration))
        
        # Previous claims (Poisson distribution)
        previous_claims = int(np.random.poisson(0.5))
        
        # Claim amount based on type
        if claim_type == 'auto':
            base_amount = np.random.lognormal(9, 1)  # ~$8k average
        elif claim_type == 'home':
            base_amount = np.random.lognormal(10, 1.2)  # ~$22k average
        elif claim_type == 'health':
            base_amount = np.random.lognormal(8.5, 1.5)  # ~$5k average
        else:  # life
            base_amount = np.random.lognormal(11, 1)  # ~$60k average
        
        claim_amount = round(base_amount, 2)
        
        # Policy coverage (usually higher than claim)
        policy_coverage = claim_amount * np.random.uniform(1.5, 4.0)
        
        # Generate claim date (last 2 years)
        days_ago = random.randint(0, 730)
        claim_date = datetime.now() - timedelta(days=days_ago)
        
        # Policy start date (before claim)
        policy_start_date = claim_date - timedelta(days=policy_duration * 30)
        
        # Claim description based on type
        descriptions = {
            'auto': [
                'Vehicle collision on highway',
                'Rear-end accident in parking lot',
                'Side-impact collision at intersection',
                'Single vehicle accident',
                'Theft of vehicle parts',
                'Vandalism damage to vehicle',
                'Hit and run incident'
            ],
            'home': [
                'Water damage from pipe burst',
                'Fire damage to kitchen',
                'Storm damage to roof',
                'Theft of personal property',
                'Vandalism to property',
                'Tree fell on house',
                'Electrical fire damage'
            ],
            'health': [
                'Emergency room visit',
                'Surgical procedure',
                'Hospital admission',
                'Diagnostic tests and imaging',
                'Physical therapy treatment',
                'Prescription medication costs',
                'Specialist consultation'
            ],
            'life': [
                'Death benefit claim',
                'Terminal illness benefit',
                'Accidental death claim',
                'Critical illness claim'
            ]
        }
        
        claim_description = random.choice(descriptions[claim_type])
        
        # Determine risk level based on multiple factors
        risk_score = 0
        
        # High claim amount relative to coverage
        if claim_amount / policy_coverage > 0.7:
            risk_score += 3
        
        # Multiple previous claims
        if previous_claims >= 2:
            risk_score += 2
        
        # New policy (< 6 months)
        if policy_duration < 6:
            risk_score += 2
        
        # Very high claim amount
        if claim_amount > 50000:
            risk_score += 2
        
        # Young or very old customers
        if customer_age < 25 or customer_age > 70:
            risk_score += 1
        
        # Early claim after policy start
        days_since_policy = (claim_date - policy_start_date).days
        if days_since_policy < 30:
            risk_score += 3
        
        # Determine final risk level
        risk_level = 'high' if risk_score >= 4 else 'low'
        
        claim = {
            'claim_id': f'CLM-2024-{i+10000}',
            'customer_id': f'CUST-{random.randint(1000, 9999)}',
            'claim_amount': round(claim_amount, 2),
            'claim_type': claim_type,
            'claim_date': claim_date.strftime('%Y-%m-%d'),
            'policy_id': f'POL-{random.randint(100000, 999999)}',
            'customer_age': customer_age,
            'policy_duration': policy_duration,
            'policy_start_date': policy_start_date.strftime('%Y-%m-%d'),
            'policy_coverage': round(policy_coverage, 2),
            'previous_claims': previous_claims,
            'claim_description': claim_description,
            'location': random.choice(locations),
            'risk_level': risk_level
        }
        
        data.append(claim)
    
    return pd.DataFrame(data)


def main():
    """Generate and save sample data"""
    print("=" * 70)
    print(" " * 15 + "GENERATING SAMPLE CLAIMS DATA")
    print("=" * 70)
    
    # Generate training data (800 samples)
    print("\nGenerating training data...")
    train_data = generate_sample_claims(800)
    train_data.to_csv('data/sample_claims_train.csv', index=False)
    print(f"✓ Generated training data: {len(train_data)} claims")
    print(f"  - High risk: {(train_data['risk_level'] == 'high').sum()}")
    print(f"  - Low risk: {(train_data['risk_level'] == 'low').sum()}")
    
    # Generate test data (200 samples)
    print("\nGenerating test data...")
    test_data = generate_sample_claims(200)
    test_data.to_csv('data/sample_claims_test.csv', index=False)
    print(f"✓ Generated test data: {len(test_data)} claims")
    
    # Generate sample claims for API testing (unlabeled)
    print("\nGenerating API test data...")
    api_test_data = test_data.head(10).drop('risk_level', axis=1)
    api_test_data.to_csv('data/sample_claims_api_test.csv', index=False)
    print(f"✓ Generated API test data: {len(api_test_data)} claims")
    
    # Display sample
    print("\n" + "=" * 70)
    print("SAMPLE DATA PREVIEW")
    print("=" * 70)
    print(train_data.head())
    
    print("\n" + "=" * 70)
    print("DATA DISTRIBUTION BY CLAIM TYPE")
    print("=" * 70)
    print(train_data['claim_type'].value_counts())
    
    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)
    print(train_data[['claim_amount', 'customer_age', 'policy_duration']].describe())
    
    print("\n" + "=" * 70)
    print("✓ DATA GENERATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - data/sample_claims_train.csv (800 records)")
    print("  - data/sample_claims_test.csv (200 records)")
    print("  - data/sample_claims_api_test.csv (10 records)")
    print("=" * 70)


if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    main()
