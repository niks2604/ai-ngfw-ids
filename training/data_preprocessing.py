"""
Data Preprocessing for CICIDS2017 Dataset
- Load all parquet files
- Clean missing/infinite values
- Encode labels
- Save processed data
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data(data_path):
    """Load all parquet files and combine them."""
    print("Loading dataset files...")
    
    all_data = []
    files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
    
    for f in files:
        filepath = os.path.join(data_path, f)
        df = pd.read_parquet(filepath)
        
        # Extract label from filename (e.g., "Benign-Monday" -> "Benign")
        label = f.split('-')[0]
        df['attack_category'] = label
        
        all_data.append(df)
        print(f"   Loaded {f}: {len(df):,} rows")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Total: {len(combined):,} rows, {len(combined.columns)} columns")
    
    return combined

def clean_data(df):
    """Handle missing values and infinities."""
    print("\nCleaning data...")
    
    original_size = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Replace infinity with NaN
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    missing = df[numeric_cols].isnull().sum().sum()
    print(f"   Found {missing:,} missing/infinite values")
    
    # Fill missing with median
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    df.dropna(inplace=True)
    
    print(f"   Rows: {original_size:,} → {len(df):,}")
    print(f"✅ Data cleaned!")
    
    return df

def encode_labels(df):
    """Convert attack labels to numbers."""
    print("\nEncoding labels...")
    
    print("   Class distribution:")
    for label, count in df['attack_category'].value_counts().items():
        print(f"      {label}: {count:,}")
    
    # Binary label (0 = Benign, 1 = Attack)
    df['is_attack'] = (df['attack_category'] != 'Benign').astype(int)
    
    # Multi-class label
    le = LabelEncoder()
    df['attack_label'] = le.fit_transform(df['attack_category'])
    
    # Save label encoder
    os.makedirs('trained_models', exist_ok=True)
    joblib.dump(le, 'trained_models/label_encoder.joblib')
    
    print(f"\n   Label mapping:")
    for i, label in enumerate(le.classes_):
        print(f"      {i} = {label}")
    
    print(f"✅ Labels encoded!")
    
    return df, le

def get_feature_columns(df):
    """Get list of feature columns."""
    exclude_cols = ['attack_category', 'is_attack', 'attack_label', 
                    'Flow ID', 'Source IP', 'Destination IP', 
                    'Src IP', 'Dst IP', 'Timestamp', 'Label']
    
    feature_cols = [col for col in df.columns 
                    if col not in exclude_cols 
                    and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    print(f"\n✅ Using {len(feature_cols)} features for training")
    
    return feature_cols

def save_processed_data(df, feature_cols, output_path):
    """Save processed data."""
    print(f"\nSaving processed data to {output_path}...")
    
    os.makedirs(output_path, exist_ok=True)
    df.to_parquet(os.path.join(output_path, 'processed_data.parquet'), index=False)
    joblib.dump(feature_cols, os.path.join(output_path, 'feature_columns.joblib'))
    
    print(f"✅ Saved!")

def main():
    print("="*50)
    print("CICIDS2017 Data Preprocessing")
    print("="*50)
    
    # Paths - adjust if needed
    data_path = os.path.expanduser('~/sem6el/data/cicids2017/')
    output_path = os.path.expanduser('~/sem6el/data/processed/')
    
    df = load_data(data_path)
    df = clean_data(df)
    df, label_encoder = encode_labels(df)
    feature_cols = get_feature_columns(df)
    save_processed_data(df, feature_cols, output_path)
    
    print("\n" + "="*50)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*50)
    print(f"\nSummary:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Benign: {(df['is_attack']==0).sum():,}")
    print(f"   Attacks: {(df['is_attack']==1).sum():,}")

if __name__ == "__main__":
    main()