"""
Graph Neural Network for Lateral Movement Detection
"""
import torch
import torch.nn.functional as F
try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    Data = None

import numpy as np
from collections import deque

class SpatialRiskGNN(torch.nn.Module):
    """
    A lightweight GraphSAGE model to detect lateral movement
    by scoring nodes based on their connection patterns.
    """
    def __init__(self, num_node_features, hidden_channels=32):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError("torch_geometric is not installed. Run pip install torch-geometric.")
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Output a risk score per node
        risk_logits = self.out(x)
        return torch.sigmoid(risk_logits)

class DynamicGraphBuilder:
    """
    Maintains a rolling window of recent connections to build a 
    dynamic graph for GNN evaluation.
    """
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.flow_buffer = deque(maxlen=window_size)
        
    def add_flows(self, flows: list[dict], contexts: list[dict]):
        for f, c in zip(flows, contexts):
            if c and c.get('src_ip') and c.get('dst_ip'):
                self.flow_buffer.append({'features': f, 'context': c})

    def build_graph(self) -> tuple[Data | None, dict]:
        if not PYG_AVAILABLE or len(self.flow_buffer) == 0:
            return None, {}
            
        nodes = {}
        edges = []
        
        # Build node index mapping
        for item in self.flow_buffer:
            ctx = item['context']
            src = ctx.get('src_ip')
            dst = ctx.get('dst_ip')
            
            if src not in nodes:
                nodes[src] = len(nodes)
            if dst not in nodes:
                nodes[dst] = len(nodes)
                
            edges.append([nodes[src], nodes[dst]])
            
        if not edges:
            return None, {}
            
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # In a fully trained model, we would compute aggregate features per node 
        # (e.g. out-degree, avg packet size). Here we use degree as a proxy feature.
        num_nodes = len(nodes)
        
        # Compute node degrees
        degrees = torch.zeros(num_nodes, dtype=torch.float)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
            
        # [degree, dummy1, dummy2, dummy3]
        x = torch.zeros((num_nodes, 4), dtype=torch.float)
        x[:, 0] = degrees
        
        return Data(x=x, edge_index=edge_index), nodes

class GNNManager:
    def __init__(self):
        self.is_loaded = False
        self.model = None
        self.builder = DynamicGraphBuilder()
        self.node_features_dim = 4
        
    def load_model(self):
        if PYG_AVAILABLE:
            self.model = SpatialRiskGNN(num_node_features=self.node_features_dim)
            self.model.eval()
            self.is_loaded = True
            
    def compute_spatial_risk(self, flows: list[dict], contexts: list[dict]) -> list[float]:
        # Default spatial risk if GNN is not ready or no context
        default_scores = [0.0] * len(flows)
        
        if not self.is_loaded or not PYG_AVAILABLE:
            return default_scores
            
        # Ensure contexts are dicts (might be FlowContextIn objects)
        ctx_dicts = []
        for c in contexts:
            if hasattr(c, "model_dump"):
                ctx_dicts.append(c.model_dump())
            elif hasattr(c, "dict"):
                ctx_dicts.append(c.dict())
            else:
                ctx_dicts.append(c or {})
                
        # Add new flows to sliding window
        self.builder.add_flows(flows, ctx_dicts)
        
        graph_data, node_map = self.builder.build_graph()
        if graph_data is None:
            return default_scores
            
        with torch.no_grad():
            node_risks = self.model(graph_data.x, graph_data.edge_index).squeeze().numpy()
            
        # Ensure it's an array even for single node
        if node_risks.ndim == 0:
            node_risks = np.array([node_risks])
            
        scores = []
        for ctx in ctx_dicts:
            score = 0.0
            # Risk is primarily evaluated based on the source IP's behavior
            if ctx and ctx.get('src_ip') in node_map:
                idx = node_map[ctx['src_ip']]
                score = float(node_risks[idx])
            scores.append(score)
            
        return scores

# Singleton instance
gnn_manager = GNNManager()
