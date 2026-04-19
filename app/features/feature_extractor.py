"""
Feature Extractor for AI-NGFW
- Normalize features
- Select important features
- Prepare data for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os

class FeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
        
    def fit(self, df, feature_columns):
        """Fit the scaler on training data."""
        print("Fitting feature extractor...")
        
        self.feature_columns = feature_columns
        X = df[feature_columns].values
        
        # Handle any remaining infinities
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.scaler.fit(X)
        self.is_fitted = True
        
        print(f"✅ Fitted on {len(feature_columns)} features")
        
        return self
    
    def transform(self, df):
        """Transform features using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor not fitted! Call fit() first.")
        
        X = df[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit_transform(self, df, feature_columns):
        """Fit and transform in one step."""
        self.fit(df, feature_columns)
        return self.transform(df)
    
    def save(self, path):
        """Save the feature extractor."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, path)
        print(f"✅ Saved feature extractor to {path}")
    
    def load(self, path):
        """Load a saved feature extractor."""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.is_fitted = True
        print(f"✅ Loaded feature extractor from {path}")
        return self


def get_top_features(df, feature_columns, target_col='is_attack', top_n=50):
    """Get most important features using correlation."""
    print(f"\nSelecting top {top_n} features...")
    
    correlations = {}
    for col in feature_columns:
        corr = abs(df[col].corr(df[target_col]))
        if not np.isnan(corr):
            correlations[col] = corr
    
    # Sort by correlation
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:top_n]]
    
    print(f"✅ Top 10 features by correlation:")
    for feat, corr in sorted_features[:10]:
        print(f"   {feat}: {corr:.4f}")
    
    return top_features


def prepare_train_test_split(df, feature_columns, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    from sklearn.model_selection import train_test_split
    
    print(f"\nSplitting data (test_size={test_size})...")
    
    X = df[feature_columns].values
    y_binary = df['is_attack'].values
    y_multi = df['attack_label'].values
    
    # Handle infinities
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Split
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
        X, y_binary, y_multi, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_binary
    )
    
    print(f"✅ Train: {len(X_train):,} samples")
    print(f"✅ Test:  {len(X_test):,} samples")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_binary': y_train_bin,
        'y_test_binary': y_test_bin,
        'y_train_multi': y_train_multi,
        'y_test_multi': y_test_multi
    }


if __name__ == "__main__":
    # Test the feature extractor
    print("="*50)
    print("Feature Engineering")
    print("="*50)
    
    # Load processed data
    data_path = os.path.expanduser('~/sem6el/data/processed/')
    
    print("\nLoading processed data...")
    df = pd.read_parquet(os.path.join(data_path, 'processed_data.parquet'))
    feature_columns = joblib.load(os.path.join(data_path, 'feature_columns.joblib'))
    
    print(f"   Loaded {len(df):,} rows, {len(feature_columns)} features")
    
    # Get top features
    top_features = get_top_features(df, feature_columns, top_n=50)
    
    # Create feature extractor
    extractor = FeatureExtractor()
    X_scaled = extractor.fit_transform(df, top_features)
    
    print(f"\n✅ Scaled features shape: {X_scaled.shape}")
    
    # Save feature extractor
    extractor.save(os.path.expanduser('~/sem6el/trained_models/feature_extractor.joblib'))
    
    # Prepare train/test split
    splits = prepare_train_test_split(df, top_features)
    
    # Save splits for training
    split_path = os.path.expanduser('~/sem6el/data/processed/')
    joblib.dump(splits, os.path.join(split_path, 'train_test_splits.joblib'))
    joblib.dump(top_features, os.path.join(split_path, 'top_features.joblib'))
    
    print("\n" + "="*50)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*50)
    print(f"\nSaved:")
    print(f"   - Feature extractor")
    print(f"   - Train/test splits")
    print(f"   - Top {len(top_features)} features")