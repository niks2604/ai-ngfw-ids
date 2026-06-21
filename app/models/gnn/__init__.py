"""Graph Neural Network layer for AI-NGFW.

This package adds a topology-aware second opinion on top of the
ensemble. The ensemble decides per-flow from row features; the GNN
decides from *who is talking to whom*, which is the only signal that
exposes coordinated patterns like DDoS, horizontal scans, or C2
beacons. It is only consulted for INSPECT decisions where the
ensemble is uncertain, so it does not slow down the hot path.

Modules
-------
- graph_builder: rolling 60s flow window -> PyTorch Geometric Data
- gnn_model:     two-headed GAT (per-node + whole-graph classifiers)
- gnn_detector:  inference wrapper used by the API
"""

from app.models.gnn.gnn_detector import GNNDetector

__all__ = ["GNNDetector"]
