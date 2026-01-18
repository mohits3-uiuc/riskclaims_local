"""
Quick Start Script - Complete Setup in One Command

This script automates the complete setup:
1. Check/install dependencies
2. Generate sample data
3. Train the model (all 10 pipeline stages)
4. Test predictions
"""

import subprocess
import sys
import os


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def print_step(step_num, total_steps, description):
    """Print step header"""
    print(f"\n[STEP {step_num}/{total_steps}] {description}")
    print("-" * 70)


def check_dependencies():
    """Check if required packages are installed"""
    print_step(1, 4, "CHECKING DEPENDENCIES")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }
    
    missing = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} installed")
        except ImportError:
            missing.append(package_name)
            print(f"❌ {package_name} NOT installed")
    
    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
        print("✓ Dependencies installed successfully")
    else:
        print("\n✓ All dependencies already installed")
    
    return True


def run_script(script_name, description):
    """Run a Python script and handle errors"""
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run {script_name}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main execution"""
    print_header("RISK CLAIMS LOCAL PIPELINE - QUICK START")
    print(f"\nThis will set up the complete ML pipeline on your local machine.")
    print(f"Expected time: 30-60 seconds\n")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Failed to install dependencies")
        sys.exit(1)
    
    # Step 2: Generate sample data
    print_step(2, 4, "GENERATING SAMPLE DATA")
    if not run_script("generate_sample_data.py", "Generate sample claims data"):
        sys.exit(1)
    
    # Step 3: Train model
    print_step(3, 4, "TRAINING MODEL (ALL 10 PIPELINE STAGES)")
    if not run_script("pipeline.py", "Train ML model"):
        sys.exit(1)
    
    # Step 4: Test model
    print_step(4, 4, "TESTING MODEL PREDICTIONS")
    if not run_script("test_pipeline.py", "Test model predictions"):
        sys.exit(1)
    
    # Success summary
    print_header("SETUP COMPLETE!")
    
    print("\n✅ Your local ML pipeline is ready to use!")
    
    print("\n📊 Generated Files:")
    print("   Data:")
    print("     - data/sample_claims_train.csv (800 records)")
    print("     - data/sample_claims_test.csv (200 records)")
    print("     - data/sample_claims_api_test.csv (10 records)")
    
    print("\n   Model:")
    print("     - models/risk_model.pkl")
    print("     - models/scaler.pkl")
    print("     - models/encoders.pkl")
    print("     - models/metadata.json")
    
    print("\n   Results:")
    print("     - data/batch_prediction_results.csv")
    
    # Display model performance
    if os.path.exists('models/metadata.json'):
        import json
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("\n🎯 Model Performance:")
        print(f"     Accuracy:  {metadata['metrics']['accuracy']:.4f}")
        print(f"     Precision: {metadata['metrics']['precision']:.4f}")
        print(f"     Recall:    {metadata['metrics']['recall']:.4f}")
        print(f"     F1 Score:  {metadata['metrics']['f1']:.4f}")
    
    print("\n📚 Next Steps:")
    print("   1. Review README.md for detailed documentation")
    print("   2. Check models/metadata.json for full metrics")
    print("   3. Try custom predictions:")
    print("      python -c \"from test_pipeline import RiskClaimsPredictor; p=RiskClaimsPredictor()\"")
    print("   4. Modify config/pipeline_config.json to customize")
    
    print("\n💡 Usage Examples:")
    print("   - Train again:       python pipeline.py")
    print("   - Test predictions:  python test_pipeline.py")
    print("   - Generate new data: python generate_sample_data.py")
    
    print("\n" + "=" * 70)
    print("Happy modeling! 🚀".center(70))
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
