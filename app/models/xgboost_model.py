"""
XGBoost - High Accuracy Classification
Best performing model for structured data
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        self.is_fitted = False
        
    def fit(self, X_train, y_train):
        """Train the classifier."""
        print("Training XGBoost...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"✅ XGBoost trained on {len(X_train):,} samples")
        return self
    
    def predict(self, X):
        """Return predicted class."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Return probability of attack."""
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        else:
            return 1 - proba[:, 0]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n📊 XGBoost Accuracy: {acc:.4f}")
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
        print(f"✅ Saved XGBoost to {path}")
        
    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"✅ Loaded XGBoost from {path}")
        return self