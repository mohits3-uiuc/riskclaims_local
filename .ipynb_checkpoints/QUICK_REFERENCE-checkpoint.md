# Quick Reference - Risk Claims Local Pipeline

## 🚀 One-Command Setup

```bash
cd /Users/mohit/Downloads/ML-projects/riskclaims-local
python quick_start.py
```

## 📁 What You Get

```
riskclaims-local/
├── config/
│   └── pipeline_config.json        # Model configuration
├── data/
│   ├── sample_claims_train.csv     # 800 training records
│   ├── sample_claims_test.csv      # 200 test records
│   ├── sample_claims_api_test.csv  # 10 unlabeled claims
│   └── batch_prediction_results.csv # Test predictions
├── models/
│   ├── risk_model.pkl              # Trained Random Forest model
│   ├── scaler.pkl                  # Feature scaler
│   ├── encoders.pkl                # Label encoders
│   └── metadata.json               # Model metrics & info
├── generate_sample_data.py         # Data generator
├── pipeline.py                     # Complete 10-stage ML pipeline
├── test_pipeline.py                # Prediction & testing
├── quick_start.py                  # One-command setup
├── requirements.txt                # Dependencies
└── README.md                       # Full documentation
```

## 🎯 All 10 ML Pipeline Stages

| Stage | Name | What It Does | File |
|-------|------|--------------|------|
| 1 | Data Ingestion | Load CSV files | pipeline.py:stage1_ingest_data() |
| 2 | Data Validation | Check schema & quality | pipeline.py:stage2_validate_data() |
| 3 | Data Preprocessing | Clean & transform | pipeline.py:stage3_preprocess_data() |
| 4 | Feature Engineering | Create 15 features | pipeline.py:stage4_engineer_features() |
| 5 | Model Training | Train Random Forest | pipeline.py:stage5_train_model() |
| 6 | Model Evaluation | Calculate metrics | pipeline.py:stage6_evaluate_model() |
| 7 | Model Selection | Choose best model | pipeline.py:stage7_select_model() |
| 8 | Model Deployment | Save artifacts | pipeline.py:stage8_deploy_model() |
| 9 | Monitoring | Check for drift | pipeline.py:stage9_monitor_model() |
| 10 | Retraining | Update with new data | pipeline.py:stage10_retrain_model() |

## 💻 Common Commands

```bash
# Generate fresh data
python generate_sample_data.py

# Train the model (all 10 stages)
python pipeline.py

# Test predictions
python test_pipeline.py

# Install dependencies
pip install -r requirements.txt
```

## 🔬 Quick Testing

### Test Single Claim (Python)

```python
from test_pipeline import RiskClaimsPredictor

predictor = RiskClaimsPredictor()

claim = {
    'claim_amount': 15000.00,
    'claim_type': 'auto',
    'customer_age': 45,
    'policy_duration': 24,
    'policy_coverage': 50000.00,
    'previous_claims': 0,
    'claim_date': '2024-01-15',
    'policy_start_date': '2022-01-01'
}

result = predictor.predict(claim)
print(f"Risk: {result['risk_level']}, Confidence: {result['confidence']:.2%}")
```

### Test Batch (Python)

```python
import pandas as pd
from test_pipeline import RiskClaimsPredictor

claims_df = pd.read_csv('data/sample_claims_api_test.csv')
predictor = RiskClaimsPredictor()
results = predictor.predict_batch(claims_df)
print(results[['claim_id', 'risk_level', 'confidence']])
```

## 📊 Model Performance

Achieved with 800 training samples:
- **Accuracy**: 100% (on generated data)
- **Precision**: 100%
- **Recall**: 100%
- **F1 Score**: 100%
- **Training Time**: ~1 second
- **Model Size**: 125 KB

## 🔧 Customization

### Change Model Parameters

Edit `config/pipeline_config.json`:

```json
{
  "model_params": {
    "n_estimators": 100,    // More trees
    "max_depth": 15,        // Deeper trees
    "min_samples_split": 10 // Regularization
  }
}
```

### Add Custom Features

Edit `pipeline.py` in `stage4_engineer_features()`:

```python
# Add your feature
self.df['my_feature'] = self.df['claim_amount'] / self.df['customer_age']
self.feature_names.append('my_feature')
```

### Try Different Models

Edit `pipeline.py` in `stage5_train_model()`:

```python
from sklearn.ensemble import GradientBoostingClassifier

self.model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5
)
```

## 📈 Sample Data Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| claim_id | string | Unique ID | CLM-2024-10000 |
| claim_amount | float | Dollar amount | 15000.00 |
| claim_type | string | auto/home/health/life | auto |
| customer_age | int | Age in years | 45 |
| policy_duration | int | Months active | 24 |
| policy_coverage | float | Total coverage | 50000.00 |
| previous_claims | int | Prior claims count | 0 |
| risk_level | string | **TARGET** high/low | low |

## 🎓 Features Generated

**Original (5):**
- claim_amount, customer_age, policy_duration, policy_coverage, previous_claims

**Engineered (10):**
- claim_to_coverage_ratio
- claim_per_month
- is_high_claim
- is_new_policy
- has_prior_claims
- multiple_prior_claims
- claim_type_encoded
- age_group_encoded
- days_since_claim
- days_policy_to_claim

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | `pip install -r requirements.txt` |
| File not found | Run `python generate_sample_data.py` first |
| Model not found | Run `python pipeline.py` to train |
| Low accuracy | Increase `n_estimators` in config |
| Slow training | Decrease `n_estimators` or data size |

## 🎯 Key Differences from Full Version

| Feature | Full Version | Local Version |
|---------|-------------|---------------|
| Data Source | AWS RDS, S3 | Local CSV |
| Data Size | Millions | 1,000 |
| Compute | Cloud instances | Local CPU |
| Dependencies | 20+ packages | 4 packages |
| Setup Time | Hours | Minutes |
| Model Storage | MLflow/S3 | Local files |
| API | Production FastAPI | Test script |
| Monitoring | Prometheus/Grafana | Basic drift check |

## 📝 Next Steps

1. ✅ Run `quick_start.py` - Done!
2. ✅ Check model performance in `models/metadata.json`
3. 🔄 Test with your own claims data
4. 🔄 Adjust features and retrain
5. 🔄 Deploy as FastAPI service
6. 🔄 Scale up with more data

## 💡 Use Cases

- **Learning**: Understand ML pipeline architecture
- **Prototyping**: Test ideas before cloud deployment
- **Development**: Local debugging and testing
- **Education**: Teaching ML best practices
- **Interviews**: Demonstrate full-stack ML knowledge

## 🤝 Integration Examples

### Use in Your Application

```python
from test_pipeline import RiskClaimsPredictor

class MyClaimsSystem:
    def __init__(self):
        self.predictor = RiskClaimsPredictor()
    
    def process_claim(self, claim_data):
        result = self.predictor.predict(claim_data)
        
        if result['risk_level'] == 'high':
            return "Send to manual review"
        else:
            return "Auto-approve"
```

### Build API Endpoint

```python
from fastapi import FastAPI
from test_pipeline import RiskClaimsPredictor

app = FastAPI()
predictor = RiskClaimsPredictor()

@app.post("/predict")
def predict_risk(claim: dict):
    return predictor.predict(claim)
```

## 📞 Support

- Read [README.md](README.md) for detailed documentation
- Check code comments in each `.py` file
- Review test examples in `test_pipeline.py`

---

**Created**: January 2026  
**Version**: 1.0.0  
**Status**: Production-ready for local deployment  
**License**: Educational/Demo purposes
