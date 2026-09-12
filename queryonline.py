import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

class FiLMQueryModel(nn.Module):
    def __init__(self, node_dim=768, hidden_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        query_dim = self.encoder.config.hidden_size
        self.query_projection = nn.Linear(query_dim, node_dim)
        
        self.film = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * node_dim)
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        raw_query = out.last_hidden_state[:, 0, :]
        e_q = self.query_projection(raw_query)
        
        params = self.film(raw_query)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        gamma = 1.0 + gamma
        
        return e_q, gamma, beta
