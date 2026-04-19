"""
Random Forest - Multi-class Classification
Classifies attacks into categories (DDoS, Botnet, etc.)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.is_fitted = False
        
    def fit(self, X_train, y_train):
        """Train the classifier."""
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"✅ Random Forest trained on {len(X_train):,} samples")
        return self
    
    def predict(self, X):
        """Return predicted class."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Return probability of attack (class 1)."""
        proba = self.model.predict_proba(X)
        # Return probability of attack class
        if proba.shape[1] == 2:
            return proba[:, 1]  # Binary: return attack probability
        else:
            # Multi-class: return max non-benign probability
            return 1 - proba[:, 0]  # Assuming 0 is benign
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n📊 Random Forest Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        return acc
    
    def get_feature_importance(self, feature_names):
        """Get feature importances."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n🔝 Top 10 Important Features:")
        for i in range(min(10, len(feature_names))):
            print(f"   {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return dict(zip(feature_names, importances))
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"✅ Saved Random Forest to {path}")
        
    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"✅ Loaded Random Forest from {path}")
        return self