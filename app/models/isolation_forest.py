"""
Isolation Forest - Anomaly Detection
Detects zero-day attacks by finding "weird" traffic
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class IsolationForestModel:
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        
    def fit(self, X_train):
        """Train on normal traffic only."""
        print("Training Isolation Forest...")
        self.model.fit(X_train)
        self.is_fitted = True
        print(f"✅ Isolation Forest trained on {len(X_train):,} samples")
        return self
    
    def predict(self, X):
        """Returns: 1 = normal, -1 = anomaly"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Return anomaly score (0-1, higher = more anomalous)."""
        # score_samples returns negative values, more negative = more anomalous
        scores = self.model.score_samples(X)
        # Convert to 0-1 range (1 = anomaly)
        proba = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return proba
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"✅ Saved Isolation Forest to {path}")
        
    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"✅ Loaded Isolation Forest from {path}")
        return self