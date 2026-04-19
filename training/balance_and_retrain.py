"""
Balance the dataset and retrain models
This creates a 50-50 split for fair training
"""

import os
import sys
sys.path.append(os.path.expanduser('~/sem6el'))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import time

def main():
    print("="*60)
    print("🔄 BALANCING DATASET & RETRAINING")
    print("="*60)
    
    # Paths
    data_path = os.path.expanduser('~/sem6el/data/processed/')
    model_path = os.path.expanduser('~/sem6el/trained_models/')
    
    # Load data
    print("\n📂 Loading data...")
    splits = joblib.load(os.path.join(data_path, 'train_test_splits.joblib'))
    
    X_train_full = splits['X_train']
    y_train_full = splits['y_train_binary']
    X_test_full = splits['X_test']
    y_test_full = splits['y_test_binary']
    
    print(f"   Original train: {len(X_train_full):,} samples")
    print(f"   Benign: {(y_train_full==0).sum():,} | Attack: {(y_train_full==1).sum():,}")
    
    # Balance by undersampling attacks
    print("\n⚖️  Balancing dataset...")
    
    benign_idx = np.where(y_train_full == 0)[0]
    attack_idx = np.where(y_train_full == 1)[0]
    
    n_benign = len(benign_idx)
    
    # Randomly sample same number of attacks as benign
    np.random.seed(42)
    attack_idx_sampled = np.random.choice(attack_idx, size=n_benign, replace=False)
    
    # Combine
    balanced_idx = np.concatenate([benign_idx, attack_idx_sampled])
    np.random.shuffle(balanced_idx)
    
    X_train = X_train_full[balanced_idx]
    y_train = y_train_full[balanced_idx]
    
    print(f"   Balanced train: {len(X_train):,} samples")
    print(f"   Benign: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.1f}%)")
    print(f"   Attack: {(y_train==1).sum():,} ({(y_train==1).mean()*100:.1f}%)")
    
    # Balance test set too
    benign_idx_test = np.where(y_test_full == 0)[0]
    attack_idx_test = np.where(y_test_full == 1)[0]
    n_benign_test = len(benign_idx_test)
    attack_idx_test_sampled = np.random.choice(attack_idx_test, size=n_benign_test, replace=False)
    balanced_idx_test = np.concatenate([benign_idx_test, attack_idx_test_sampled])
    
    X_test = X_test_full[balanced_idx_test]
    y_test = y_test_full[balanced_idx_test]
    
    print(f"\n   Balanced test: {len(X_test):,} samples")
    print(f"   Benign: {(y_test==0).sum():,} | Attack: {(y_test==1).sum():,}")
    
    # Scale
    print("\n📐 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_path, 'scaler.joblib'))
    
    # ============================================
    # Train Models
    # ============================================
    
    # 1. Isolation Forest
    print("\n" + "="*60)
    print("1️⃣  ISOLATION FOREST")
    print("="*60)
    
    start = time.time()
    X_normal = X_train_scaled[y_train == 0]
    
    # Sample for speed
    if len(X_normal) > 50000:
        idx = np.random.choice(len(X_normal), 50000, replace=False)
        X_normal = X_normal[idx]
    
    print(f"   Training on {len(X_normal):,} normal samples...")
    
    iso_model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
    iso_model.fit(X_normal)
    joblib.dump(iso_model, os.path.join(model_path, 'isolation_forest.joblib'))
    
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    
    # 2. Random Forest
    print("\n" + "="*60)
    print("2️⃣  RANDOM FOREST")
    print("="*60)
    
    start = time.time()
    
    # Sample for speed
    if len(X_train_scaled) > 100000:
        idx = np.random.choice(len(X_train_scaled), 100000, replace=False)
        X_rf = X_train_scaled[idx]
        y_rf = y_train[idx]
    else:
        X_rf = X_train_scaled
        y_rf = y_train
    
    print(f"   Training on {len(X_rf):,} samples...")
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf_model.fit(X_rf, y_rf)
    joblib.dump(rf_model, os.path.join(model_path, 'random_forest.joblib'))
    
    # Evaluate
    y_pred_rf = rf_model.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    print(f"   📊 Accuracy: {acc_rf*100:.2f}%")
    
    # 3. XGBoost
    print("\n" + "="*60)
    print("3️⃣  XGBOOST")
    print("="*60)
    
    start = time.time()
    print(f"   Training on {len(X_rf):,} samples...")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_rf, y_rf)
    joblib.dump(xgb_model, os.path.join(model_path, 'xgboost.joblib'))
    
    # Evaluate
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    print(f"   📊 Accuracy: {acc_xgb*100:.2f}%")
    
    # ============================================
    # Final Evaluation
    # ============================================
    print("\n" + "="*60)
    print("📊 FINAL EVALUATION ON BALANCED TEST SET")
    print("="*60)
    
    print("\n🌲 Random Forest:")
    print(classification_report(y_test, y_pred_rf, target_names=['Benign', 'Attack']))
    
    print("\n⚡ XGBoost:")
    print(classification_report(y_test, y_pred_xgb, target_names=['Benign', 'Attack']))
    
    # Save balanced splits for future use
    balanced_splits = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_binary': y_train,
        'y_test_binary': y_test
    }
    joblib.dump(balanced_splits, os.path.join(data_path, 'balanced_splits.joblib'))
    
    print("\n" + "="*60)
    print("✅ RETRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Saved balanced data to: {data_path}balanced_splits.joblib")

if __name__ == "__main__":
    main()