"""
Fix data issues and retrain models properly
"""

import os
import sys
sys.path.append(os.path.expanduser('~/sem6el'))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler  # Better for outliers!
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("🔧 FIXING DATA & RETRAINING")
    print("="*60)
    
    # Paths
    data_path = os.path.expanduser('~/sem6el/data/processed/')
    model_path = os.path.expanduser('~/sem6el/trained_models/')
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_parquet(os.path.join(data_path, 'processed_data.parquet'))
    print(f"   Original: {len(df):,} samples")
    
    # ============================================
    # FIX 1: Proper binary labels
    # ============================================
    print("\n1️⃣ Fixing labels...")
    df['is_attack'] = (df['attack_category'] != 'Benign').astype(int)
    print(f"   Benign: {(df['is_attack']==0).sum():,}")
    print(f"   Attack: {(df['is_attack']==1).sum():,}")
    
    # ============================================
    # FIX 2: Get numeric features only
    # ============================================
    exclude_cols = ['attack_category', 'is_attack', 'attack_label']
    feature_cols = [col for col in df.columns 
                    if col not in exclude_cols 
                    and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    print(f"\n2️⃣ Using {len(feature_cols)} features")
    
    # ============================================
    # FIX 3: Handle outliers and bad values
    # ============================================
    print("\n3️⃣ Cleaning outliers...")
    
    X = df[feature_cols].values.copy()
    y = df['is_attack'].values.copy()
    
    # Replace infinities
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Replace negative values with 0 (flow durations can't be negative)
    X[X < 0] = 0
    
    # Clip extreme outliers (cap at 99.9th percentile)
    for i in range(X.shape[1]):
        p999 = np.percentile(X[:, i], 99.9)
        X[:, i] = np.clip(X[:, i], 0, p999)
    
    print("   ✅ Outliers clipped to 99.9th percentile")
    
    # ============================================
    # FIX 4: Balance the dataset
    # ============================================
    print("\n4️⃣ Balancing dataset...")
    
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    
    n_samples = min(len(benign_idx), len(attack_idx))
    
    np.random.seed(42)
    benign_sampled = np.random.choice(benign_idx, size=n_samples, replace=False)
    attack_sampled = np.random.choice(attack_idx, size=n_samples, replace=False)
    
    balanced_idx = np.concatenate([benign_sampled, attack_sampled])
    np.random.shuffle(balanced_idx)
    
    X_balanced = X[balanced_idx]
    y_balanced = y[balanced_idx]
    
    print(f"   Balanced: {len(X_balanced):,} samples (50/50)")
    
    # ============================================
    # FIX 5: Train/test split
    # ============================================
    print("\n5️⃣ Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=0.2,
        random_state=42,
        stratify=y_balanced
    )
    
    print(f"   Train: {len(X_train):,}")
    print(f"   Test: {len(X_test):,}")
    
    # ============================================
    # FIX 6: Use RobustScaler (handles outliers better)
    # ============================================
    print("\n6️⃣ Scaling with RobustScaler...")
    
    scaler = RobustScaler()  # Uses median instead of mean - robust to outliers!
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(model_path, 'scaler.joblib'))
    joblib.dump(feature_cols, os.path.join(model_path, 'feature_columns.joblib'))
    
    # Check scaling worked
    print(f"   Train mean: {X_train_scaled.mean():.4f}")
    print(f"   Train std: {X_train_scaled.std():.4f}")
    
    # ============================================
    # TRAIN MODELS
    # ============================================
    
    # 1. Random Forest
    print("\n" + "="*60)
    print("1️⃣  RANDOM FOREST")
    print("="*60)
    
    start = time.time()
    
    # Use subset for faster training (300k samples)
    n_train = min(300000, len(X_train_scaled))
    idx = np.random.choice(len(X_train_scaled), n_train, replace=False)
    X_rf = X_train_scaled[idx]
    y_rf = y_train[idx]
    
    print(f"   Training on {n_train:,} samples...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_rf, y_rf)
    joblib.dump(rf_model, os.path.join(model_path, 'random_forest.joblib'))
    
    y_pred_rf = rf_model.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    print(f"   📊 Accuracy: {acc_rf*100:.2f}%")
    
    # 2. XGBoost
    print("\n" + "="*60)
    print("2️⃣  XGBOOST")
    print("="*60)
    
    start = time.time()
    print(f"   Training on {n_train:,} samples...")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_rf, y_rf)
    joblib.dump(xgb_model, os.path.join(model_path, 'xgboost.joblib'))
    
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    print(f"   📊 Accuracy: {acc_xgb*100:.2f}%")
    
    # 3. Isolation Forest
    print("\n" + "="*60)
    print("3️⃣  ISOLATION FOREST")
    print("="*60)
    
    start = time.time()
    
    X_normal = X_train_scaled[y_train == 0]
    if len(X_normal) > 50000:
        idx_n = np.random.choice(len(X_normal), 50000, replace=False)
        X_normal = X_normal[idx_n]
    
    print(f"   Training on {len(X_normal):,} normal samples...")
    
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X_normal)
    joblib.dump(iso_model, os.path.join(model_path, 'isolation_forest.joblib'))
    
    # Evaluate IF
    iso_pred = iso_model.predict(X_test_scaled)
    iso_pred_binary = (iso_pred == -1).astype(int)
    acc_iso = accuracy_score(y_test, iso_pred_binary)
    
    print(f"   ⏱️  Time: {time.time()-start:.1f}s")
    print(f"   📊 Accuracy: {acc_iso*100:.2f}%")
    
    # ============================================
    # SAVE TEST DATA
    # ============================================
    balanced_splits = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_binary': y_train,
        'y_test_binary': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled
    }
    joblib.dump(balanced_splits, os.path.join(data_path, 'balanced_splits.joblib'))
    
    # ============================================
    # FINAL RESULTS
    # ============================================
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    print("\n🌲 Random Forest:")
    print(classification_report(y_test, y_pred_rf, target_names=['Benign', 'Attack']))
    
    print("\n⚡ XGBoost:")
    print(classification_report(y_test, y_pred_xgb, target_names=['Benign', 'Attack']))
    
    print("\n🔍 Isolation Forest:")
    print(classification_report(y_test, iso_pred_binary, target_names=['Benign', 'Attack']))
    
    # Summary
    print("\n" + "="*60)
    print("📈 SUMMARY")
    print("="*60)
    print(f"   Random Forest:     {acc_rf*100:.2f}%")
    print(f"   XGBoost:           {acc_xgb*100:.2f}%")
    print(f"   Isolation Forest:  {acc_iso*100:.2f}%")
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()