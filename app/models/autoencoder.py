"""
Autoencoder - Zero-Day Detection (Optimized for Speed)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

class AutoencoderNet(nn.Module):
    def __init__(self, input_dim):
        super(AutoencoderNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderModel:
    def __init__(self, input_dim=50, epochs=5, batch_size=1024, lr=0.001):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.threshold = None
        self.device = torch.device('cpu')
        self.is_fitted = False
        self.min_val = None
        self.max_val = None
        
    def fit(self, X_train):
        """Train on normal traffic only."""
        print(f"   Training on {len(X_train):,} samples...")
        
        # Sample if too large
        if len(X_train) > 50000:
            print(f"   Sampling 50,000 for speed...")
            indices = np.random.choice(len(X_train), 50000, replace=False)
            X_train = X_train[indices]
        
        # Normalize to 0-1
        self.min_val = X_train.min(axis=0)
        self.max_val = X_train.max(axis=0)
        X_norm = (X_train - self.min_val) / (self.max_val - self.min_val + 1e-10)
        
        # Create model
        self.model = AutoencoderNet(self.input_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Convert to tensor once
        X_tensor = torch.FloatTensor(X_norm)
        n_samples = len(X_tensor)
        
        # Training loop (manual batching - faster than DataLoader)
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            n_batches = 0
            
            # Shuffle indices
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                batch = X_tensor[batch_idx]
                
                optimizer.zero_grad()
                output = self.model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            print(f"   Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        # Set threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor).numpy()
            errors = np.mean((X_norm - reconstructed) ** 2, axis=1)
            self.threshold = np.percentile(errors, 95)
        
        self.is_fitted = True
        print(f"   ✅ Autoencoder trained, threshold: {self.threshold:.6f}")
        return self
    
    def predict_proba(self, X):
        """Return anomaly score (higher = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted!")
        
        X_norm = (X - self.min_val) / (self.max_val - self.min_val + 1e-10)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm)
            reconstructed = self.model(X_tensor).numpy()
        
        errors = np.mean((X_norm - reconstructed) ** 2, axis=1)
        scores = np.clip(errors / (self.threshold * 2), 0, 1)
        return scores
    
    def predict(self, X):
        """Return 1 for anomaly, 0 for normal."""
        return (self.predict_proba(X) > 0.5).astype(int)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'threshold': self.threshold,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'input_dim': self.input_dim
        }, path)
        print(f"   ✅ Saved Autoencoder to {path}")
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint['input_dim']
        self.model = AutoencoderNet(self.input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.threshold = checkpoint['threshold']
        self.min_val = checkpoint['min_val']
        self.max_val = checkpoint['max_val']
        self.is_fitted = True
        return self