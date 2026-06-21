"""
Two-headed Graph Attention Network for AI-NGFW.

Architecture
------------
- Two stacked ``GATConv`` layers (multi-head -> averaged head) over the
  flow graph produced by :mod:`graph_builder`. GAT is preferred over a
  plain GCN here because the in-edges of a DDoS victim and the out-edges
  of a scanner have very different roles, and attention lets the model
  weight them rather than averaging them.
- **Node head**: a 2-layer MLP that emits a per-IP attack probability
  in ``[0, 1]``. The API uses this to score the endpoint(s) of an
  INSPECT flow.
- **Graph head**: global mean+max pooling -> MLP, emits a single
  ``[0, 1]`` score for the whole window (used by ``/network/graph``).

The module degrades gracefully when torch / torch-geometric is not
installed — importing the file does not pull torch in; only
:class:`GATDetectorModel` does. This lets the rest of the project
import :mod:`graph_builder` and the topology-hint path even on a
torch-free deployment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.models.gnn.graph_builder import EDGE_FEATURE_DIM, NODE_FEATURE_DIM

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import torch


def _require_torch():
    try:
        import torch  # noqa: F401
        from torch_geometric.nn import GATConv  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "torch and torch-geometric are required for the GNN model. "
            "Install with: pip install torch torch-geometric"
        ) from e


class GATDetectorModel:
    """Factory for the GAT module.

    Wrapped in a factory so importing this file is free on
    torch-less deployments — the actual ``nn.Module`` is only built
    when :meth:`build` is called.
    """

    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout

    def build(self):
        _require_torch()
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

        node_dim = self.node_dim
        edge_dim = self.edge_dim
        hidden_dim = self.hidden_dim
        heads = self.heads
        dropout = self.dropout

        class GATDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.gat1 = GATConv(
                    node_dim, hidden_dim, heads=heads,
                    edge_dim=edge_dim, dropout=dropout,
                )
                self.gat2 = GATConv(
                    hidden_dim * heads, hidden_dim, heads=1,
                    concat=False, edge_dim=edge_dim, dropout=dropout,
                )

                # Per-node attack head -> sigmoid in forward().
                self.node_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )

                # Whole-graph head: mean + max pooled embedding.
                self.graph_head = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x, edge_index, edge_attr=None, batch=None):
                h = self.gat1(x, edge_index, edge_attr=edge_attr)
                h = F.elu(h)
                h = F.dropout(h, p=dropout, training=self.training)
                h = self.gat2(h, edge_index, edge_attr=edge_attr)
                h = F.elu(h)

                node_logits = self.node_head(h).squeeze(-1)

                if batch is None:
                    # Single-graph forward: synthesise a batch index.
                    batch = torch.zeros(h.size(0), dtype=torch.long,
                                        device=h.device)
                pooled = torch.cat(
                    [global_mean_pool(h, batch), global_max_pool(h, batch)],
                    dim=-1,
                )
                graph_logits = self.graph_head(pooled).squeeze(-1)

                return node_logits, graph_logits

            @torch.no_grad()
            def predict(self, data):
                """Return per-node and graph probabilities for one graph."""
                self.eval()
                node_logits, graph_logits = self.forward(
                    data.x, data.edge_index,
                    edge_attr=getattr(data, "edge_attr", None),
                )
                node_p = torch.sigmoid(node_logits).cpu().numpy()
                graph_p = torch.sigmoid(graph_logits).cpu().numpy()
                return node_p, float(graph_p.reshape(-1)[0]) if graph_p.size else 0.0

        return GATDetector()


def save_model(model: "torch.nn.Module", path: str) -> None:
    """Persist GNN weights to disk."""
    _require_torch()
    import os
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str, **factory_kwargs):
    """Rebuild the architecture and load weights from ``path``."""
    _require_torch()
    import torch

    model = GATDetectorModel(**factory_kwargs).build()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
