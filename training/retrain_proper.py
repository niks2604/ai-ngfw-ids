"""
Proper retraining with full data and better settings
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
    print("🔄 PROPER RETRAINING")
    print("="*60)
    
    # Load original processed data
    data_path = os.path.expanduser('~/sem6el/data/processed/')
    model_path = os.path.expanduser('~/sem6el/trained_models/')
    
    print("\n📂 Loading processed data...")
    df = pd.read_parquet(os.path.join(data_path, 'processed_data.parquet'))
    
    print(f"   Total samples: {len(df):,}")
    print(f"   Benign: {(df['is_attack']==0).sum():,}")
    print(f"   Attack: {(df['is_attack']==1).sum():,}")
    
    # Get all numeric features (not just top 50)
    exclude_cols = ['attack_category', 'is_attack', 'attack_label']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    print(f"   Using {len(feature_cols)} features")
    
    # Prepare X and y
    X = df[feature_cols].values
    y = df['is_attack'].values
    
    # Handle infinities
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Balance the data
    print("\n⚖️  Balancing dataset...")
    
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    
    n_benign = len(benign_idx)
    n_attack = len(attack_idx)
    
    # Use whichever is smaller as the sample size
    n_samples = min(n_benign, n_attack)
    
    np.random.seed(42)
    benign_sampled = np.random.choice(benign_idx, size=n_samples, replace=False)
    attack_sampled = np.random.choice(attack_idx, size=n_samples, replace=False)
    
    balanced_idx = np.concatenate([benign_sampled, attack_sampled])
    np.random.shuffle(balanced_idx)
    
    X_balanced = X[balanced_idx]
    y_balanced = y[balanced_idx]
    
    print(f"   Balanced samples: {len(X_balanced):,}")
    print(f"   Benign: {(y_balanced==0).sum():,} (50%)")
    print(f"   Attack: {(y_balanced==1).sum():,} (50%)")
    
    # Train/test split
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, 
        test_size=0.2, 
        random_state=42,
        stratify=y_balanced
    )
    
    print(f"   Train: {len(X_train):,}")
    print(f"   Test: {len(X_test):,}")
    
    # Scale
    print("\n📐 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_path, 'scaler.joblib'))
    
    # Save feature columns
    joblib.dump(feature_cols, os.path.join(model_path, 'feature_columns.joblib'))
    
    # ============================================
    # 1. Random Forest (Better Params)
    # ============================================
    print("\n" + "="*60)
    print("1️⃣  RANDOM FOREST")
    print("="*60)
    
    start = time.time()
    print(f"   Training on {len(X_train):,} samples...")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,       # More trees
        max_depth=30,           # Deeper trees
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train_scaled, y_train)
    joblib.dump(rf_model, os.path.join(model_path, 'random_forest.joblib'))
    
    y_pred_rf = rf_model.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    print(f"   📊 Accuracy: {acc_rf*100:.2f}%")
    
    # ============================================
    # 2. XGBoost (Better Params)
    # ============================================
    print("\n" + "="*60)
    print("2️⃣  XGBOOST")
    print("="*60)
    
    start = time.time()
    print(f"   Training on {len(X_train):,} samples...")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    joblib.dump(xgb_model, os.path.join(model_path, 'xgboost.joblib'))
    
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    print(f"   📊 Accuracy: {acc_xgb*100:.2f}%")
    
    # ============================================
    # 3. Isolation Forest
    # ============================================
    print("\n" + "="*60)
    print("3️⃣  ISOLATION FOREST")
    print("="*60)
    
    start = time.time()
    X_normal = X_train_scaled[y_train == 0]
    print(f"   Training on {len(X_normal):,} normal samples...")
    
    iso_model = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X_normal)
    joblib.dump(iso_model, os.path.join(model_path, 'isolation_forest.joblib'))
    
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    
    # ============================================
    # Save test data for ensemble evaluation
    # ============================================
    balanced_splits = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_binary': y_train,
        'y_test_binary': y_test,
        'feature_cols': feature_cols
    }
    joblib.dump(balanced_splits, os.path.join(data_path, 'balanced_splits.joblib'))
    
    # ============================================
    # Final Results
    # ============================================
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    print("\n🌲 Random Forest:")
    print(classification_report(y_test, y_pred_rf, target_names=['Benign', 'Attack']))
    
    print("\n⚡ XGBoost:")
    print(classification_report(y_test, y_pred_xgb, target_names=['Benign', 'Attack']))
    
    # Top 10 important features
    print("\n🔝 Top 10 Important Features (Random Forest):")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"   {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    print("\n" + "="*60)
    print("✅ RETRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()