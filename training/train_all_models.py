"""
Train All ML Models for AI-NGFW
"""

import sys
import os
sys.path.append(os.path.expanduser('~/sem6el'))

import numpy as np
import joblib
import time

from app.models.isolation_forest import IsolationForestModel
from app.models.random_forest import RandomForestModel
from app.models.xgboost_model import XGBoostModel
from app.models.autoencoder import AutoencoderModel

def main():
    print("="*60)
    print("🚀 TRAINING ALL ML MODELS")
    print("="*60)
    
    # Load data
    data_path = os.path.expanduser('~/sem6el/data/processed/')
    model_path = os.path.expanduser('~/sem6el/trained_models/')
    
    print("\n📂 Loading training data...")
    splits = joblib.load(os.path.join(data_path, 'train_test_splits.joblib'))
    features = joblib.load(os.path.join(data_path, 'top_features.joblib'))
    
    X_train = splits['X_train']
    X_test = splits['X_test']
    y_train = splits['y_train_binary']
    y_test = splits['y_test_binary']
    
    print(f"   Train: {X_train.shape}")
    print(f"   Test:  {X_test.shape}")
    
    # Scale the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_path, 'scaler.joblib'))
    
    results = {}
    
    # ============================================
    # 1. Isolation Forest
    # ============================================
    print("\n" + "="*60)
    print("1️⃣  ISOLATION FOREST")
    print("="*60)
    
    start = time.time()
    
    # Train on normal traffic only
    X_normal = X_train_scaled[y_train == 0]
    print(f"   Training on {len(X_normal):,} normal samples...")
    
    iso_forest = IsolationForestModel(contamination=0.1, n_estimators=100)
    iso_forest.fit(X_normal)
    iso_forest.save(os.path.join(model_path, 'isolation_forest.joblib'))
    
    # Evaluate
    scores = iso_forest.predict_proba(X_test_scaled)
    preds = (scores > 0.5).astype(int)
    acc = (preds == y_test).mean()
    results['isolation_forest'] = acc
    
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    print(f"   📊 Accuracy: {acc:.4f}")
    
    # ============================================
    # 2. Random Forest
    # ============================================
    print("\n" + "="*60)
    print("2️⃣  RANDOM FOREST")
    print("="*60)
    
    start = time.time()
    
    rf_model = RandomForestModel(n_estimators=100, max_depth=20)
    rf_model.fit(X_train_scaled, y_train)
    rf_model.save(os.path.join(model_path, 'random_forest.joblib'))
    
    acc = rf_model.evaluate(X_test_scaled, y_test)
    results['random_forest'] = acc
    
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    
    # ============================================
    # 3. XGBoost
    # ============================================
    print("\n" + "="*60)
    print("3️⃣  XGBOOST")
    print("="*60)
    
    start = time.time()
    
    xgb_model = XGBoostModel(n_estimators=100, max_depth=6)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_model.save(os.path.join(model_path, 'xgboost.joblib'))
    
    acc = xgb_model.evaluate(X_test_scaled, y_test)
    results['xgboost'] = acc
    
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    
    # ============================================
    # 4. Autoencoder
    # ============================================
    print("\n" + "="*60)
    print("4️⃣  AUTOENCODER")
    print("="*60)
    
    start = time.time()
    
    # Train on normal traffic only
    ae_model = AutoencoderModel(input_dim=X_train_scaled.shape[1], epochs=10, batch_size=512)
    ae_model.fit(X_normal)
    ae_model.save(os.path.join(model_path, 'autoencoder.pt'))
    
    # Evaluate
    scores = ae_model.predict_proba(X_test_scaled)
    preds = (scores > 0.5).astype(int)
    acc = (preds == y_test).mean()
    results['autoencoder'] = acc
    
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    print(f"   📊 Accuracy: {acc:.4f}")
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    
    print("\n📊 Model Accuracies:")
    for model, acc in results.items():
        print(f"   {model}: {acc:.4f}")
    
    print(f"\n📁 Models saved to: {model_path}")
    print("\nFiles created:")
    for f in os.listdir(model_path):
        size = os.path.getsize(os.path.join(model_path, f)) / 1024 / 1024
        print(f"   {f}: {size:.1f} MB")

if __name__ == "__main__":
    main()