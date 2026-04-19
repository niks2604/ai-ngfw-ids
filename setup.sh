#!/bin/bash

echo "=============================================="
echo "🚀 AI-NGFW Setup Script"
echo "=============================================="

cd ~/sem6el

# Activate virtual environment
echo ""
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install OpenMP for XGBoost (Mac)
echo ""
echo "📦 Installing OpenMP (required for XGBoost)..."
brew install libomp

# Create requirements.txt
echo ""
echo "📝 Creating requirements.txt..."
cat > requirements.txt << 'REQEOF'
# Core Data Science
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.12.0

# Machine Learning
scikit-learn>=1.4.0
xgboost>=2.0.0
imbalanced-learn>=0.12.0

# Deep Learning
torch>=2.1.0

# Model Serialization
joblib>=1.3.0

# Explainability
shap>=0.44.0

# API Framework
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0

# Data Formats
pyarrow>=15.0.0

# Monitoring
prometheus-client>=0.19.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
REQEOF

# Install Python packages
echo ""
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "✅ Verifying installation..."
python3 -c "
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import torch
import shap
import fastapi

print('✅ All packages installed successfully!')
print(f'   NumPy: {np.__version__}')
print(f'   Pandas: {pd.__version__}')
print(f'   Scikit-learn: {sklearn.__version__}')
print(f'   XGBoost: {xgb.__version__}')
print(f'   PyTorch: {torch.__version__}')
print(f'   SHAP: {shap.__version__}')
print(f'   FastAPI: {fastapi.__version__}')
"

echo ""
echo "=============================================="
echo "✅ SETUP COMPLETE!"
echo "=============================================="
echo ""
