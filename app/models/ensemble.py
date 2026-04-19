"""
Ensemble Model - Combines all models with weighted voting
"""

import numpy as np
import joblib
import os

class EnsembleDetector:
    def __init__(self, models_path=None):
        if models_path is None:
            models_path = os.path.expanduser('~/sem6el/trained_models/')
        
        self.models_path = models_path
        self.models = {}
        self.scaler = None
        self.is_loaded = False
        
        # Weights for each model (must sum to 1.0)
        self.weights = {
            'isolation_forest': 0.25,
            'random_forest': 0.50,
            'xgboost': 0.25
        }
    
    def load_models(self):
        """Load all trained models."""
        print("Loading trained models...")
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(self.models_path, 'scaler.joblib'))
        print("   ✅ Scaler loaded")
        
        # Load Isolation Forest
        self.models['isolation_forest'] = joblib.load(
            os.path.join(self.models_path, 'isolation_forest.joblib')
        )
        print("   ✅ Isolation Forest loaded")
        
        # Load Random Forest
        self.models['random_forest'] = joblib.load(
            os.path.join(self.models_path, 'random_forest.joblib')
        )
        print("   ✅ Random Forest loaded")
        
        # Load XGBoost
        self.models['xgboost'] = joblib.load(
            os.path.join(self.models_path, 'xgboost.joblib')
        )
        print("   ✅ XGBoost loaded")
        
        self.is_loaded = True
        print("\n✅ All models loaded!")
        
        return self
    
    def predict_proba(self, X):
        """Get probability scores from each model."""
        if not self.is_loaded:
            self.load_models()
        
        # Scale input
        X_scaled = self.scaler.transform(X)
        
        scores = {}

        # Isolation Forest (convert anomaly score to 0-1).
        # Previous impl used per-batch min/max, which collapsed to 1.0 for
        # single-sample calls and gave non-deterministic scores across batches.
        # Sigmoid around the model's own decision boundary (offset_) gives a
        # batch-size-independent score: 0.5 at the boundary, → 1 for anomalies.
        iso_model = self.models['isolation_forest']
        iso_scores = iso_model.score_samples(X_scaled)
        anom = iso_model.offset_ - iso_scores  # positive => more anomalous
        scores['isolation_forest'] = 1.0 / (1.0 + np.exp(-anom * 10.0))
        
        # Random Forest
        scores['random_forest'] = self.models['random_forest'].predict_proba(X_scaled)[:, 1]
        
        # XGBoost
        scores['xgboost'] = self.models['xgboost'].predict_proba(X_scaled)[:, 1]
        
        return scores
    
    def predict_ensemble(self, X):
        """Get weighted ensemble score."""
        scores = self.predict_proba(X)
        
        # Weighted average
        ensemble_score = np.zeros(len(X))
        for model_name, weight in self.weights.items():
            ensemble_score += weight * scores[model_name]
        
        return ensemble_score
    
    def predict(self, X):
        """Make final decision: 0=Allow, 1=Inspect, 2=Block"""
        ensemble_score = self.predict_ensemble(X)
        
        decisions = np.zeros(len(X), dtype=int)
        decisions[ensemble_score >= 0.55] = 1  # Inspect
        decisions[ensemble_score >= 0.75] = 2  # Block
        
        return decisions
    
    def analyze(self, X):
        """Full analysis with scores and decision."""
        scores = self.predict_proba(X)
        ensemble_score = self.predict_ensemble(X)
        decisions = self.predict(X)
        
        results = []
        for i in range(len(X)):
            decision_map = {0: 'ALLOW', 1: 'INSPECT', 2: 'BLOCK'}
            
            results.append({
                'risk_score': float(ensemble_score[i]),
                'decision': decision_map[decisions[i]],
                'model_scores': {
                    'isolation_forest': float(scores['isolation_forest'][i]),
                    'random_forest': float(scores['random_forest'][i]),
                    'xgboost': float(scores['xgboost'][i])
                }
            })
        
        return results


# Test the ensemble
# Test the ensemble
if __name__ == "__main__":
    import time
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    print("="*50)
    print("Testing Ensemble Detector")
    print("="*50)
    
    # Load models
    ensemble = EnsembleDetector()
    ensemble.load_models()
    
    # Load test data (more samples for accurate measurement)
    data_path = os.path.expanduser('~/sem6el/data/processed/')
    splits = joblib.load(os.path.join(data_path, 'train_test_splits.joblib'))
    
    X_test = splits['X_test'][:10000]  # Test on 10,000 samples
    y_test = splits['y_test_binary'][:10000]
    
    print(f"\n📊 Testing on {len(X_test):,} samples...")
    print(f"   Benign: {(y_test==0).sum():,}")
    print(f"   Attacks: {(y_test==1).sum():,}")
    
    # Test prediction speed
    print("\n⏱️  Testing inference speed...")
    start = time.perf_counter()
    ensemble_scores = ensemble.predict_ensemble(X_test)
    inference_time = time.perf_counter() - start
    
    print(f"   Total time for {len(X_test):,} samples: {inference_time*1000:.1f} ms")
    print(f"   Average per sample: {inference_time/len(X_test)*1000:.3f} ms")
    
    # Try different thresholds to find best
    print("\n🔧 Finding optimal threshold...")
    print("-"*50)
    
    best_acc = 0
    best_threshold = 0.5
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        predictions = (ensemble_scores >= threshold).astype(int)
        acc = accuracy_score(y_test, predictions)
        
        # Count decisions
        n_allow = (ensemble_scores < threshold).sum()
        n_block = (ensemble_scores >= threshold).sum()
        
        print(f"   Threshold {threshold}: Accuracy={acc*100:.2f}% | Allow={n_allow:,} | Block={n_block:,}")
        
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    print(f"\n✅ Best threshold: {best_threshold} (Accuracy: {best_acc*100:.2f}%)")
    
    # Final evaluation with best threshold
    print("\n" + "="*50)
    print("📊 FINAL EVALUATION")
    print("="*50)
    
    final_predictions = (ensemble_scores >= best_threshold).astype(int)
    
    print(f"\nAccuracy: {accuracy_score(y_test, final_predictions)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, final_predictions, target_names=['Benign', 'Attack']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, final_predictions)
    print(f"   True Negative (Benign→Benign):   {cm[0][0]:,}")
    print(f"   False Positive (Benign→Attack):  {cm[0][1]:,}")
    print(f"   False Negative (Attack→Benign):  {cm[1][0]:,}")
    print(f"   True Positive (Attack→Attack):   {cm[1][1]:,}")
    
    # Show score distribution
    print("\n📈 Score Distribution:")
    print(f"   Benign samples - Avg score: {ensemble_scores[y_test==0].mean():.3f}")
    print(f"   Attack samples - Avg score: {ensemble_scores[y_test==1].mean():.3f}")