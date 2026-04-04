"""
Vector Field Network
====================
Time-conditioned MLP that parameterizes the velocity field v_θ(x, t)
for Flow Matching / Neural ODE generative models.

Architecture:
    Input: concatenation of [x, time_embedding(t)]
    → MLP with residual connections
    → Output: v ∈ R^d (same dimension as x)

The time embedding uses sinusoidal positional encoding (from Transformers),
which helps the network distinguish different time steps.
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding, similar to Transformer positional encoding.
    
    Maps scalar time t ∈ [0, 1] to a d-dimensional vector using:
        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    
    This provides a smooth, continuous representation of time that helps
    the network learn time-dependent dynamics.
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Precompute frequency scales
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values, shape (batch_size,) or (batch_size, 1)
        Returns:
            Time embedding, shape (batch_size, embed_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        
        args = t * self.freqs  # (B, half_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class ResidualBlock(nn.Module):
    """Residual MLP block with layer normalization.
    
    x → LayerNorm → Linear → SiLU → Linear → + x
    
    Using SiLU (Swish) activation, which is smooth and works well for
    continuous dynamics modeling.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VectorFieldMLP(nn.Module):
    """Time-conditioned MLP for learning vector fields v_θ(x, t).
    
    This is the neural network at the heart of Flow Matching:
    it learns to predict the velocity field that transports
    noise z ~ N(0, I) to data x ~ p_data.
    
    Architecture:
        1. Sinusoidal time embedding: t → e_t ∈ R^{time_dim}
        2. Input projection: [x; e_t] → h ∈ R^{hidden_dim}
        3. Residual blocks: h → h (N layers)
        4. Output projection: h → v ∈ R^{data_dim}
    
    Args:
        data_dim: Dimension of the data/latent space.
        hidden_dim: Hidden layer dimension.
        time_dim: Time embedding dimension.
        n_layers: Number of residual blocks.
    """
    
    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 256,
        time_dim: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        
        # Input: data + time embedding
        self.input_proj = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(n_layers)]
        )
        
        # Output projection to data dimension
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, data_dim),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field v_θ(x, t).
        
        Args:
            x: Current state, shape (batch_size, data_dim)
            t: Time, shape (batch_size,) with values in [0, 1]
            
        Returns:
            Predicted velocity v, shape (batch_size, data_dim)
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (B, time_dim)
        
        # Concatenate input with time embedding
        h = torch.cat([x, t_emb], dim=-1)  # (B, data_dim + time_dim)
        h = self.input_proj(h)  # (B, hidden_dim)
        
        # Process through residual blocks
        for block in self.blocks:
            h = block(h)
        
        # Output velocity
        v = self.output_proj(h)  # (B, data_dim)
        return v
