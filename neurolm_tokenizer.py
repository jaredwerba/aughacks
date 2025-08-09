"""
NeuroLM Real-time EEG Tokenizer
Adapted from NeuroLM repository for real-time attention metrics extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from dataclasses import dataclass
import pickle
import math
from typing import Optional, Tuple, Dict, Any


# Standard 10-20 electrode system mapping
STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'pad'
]


@dataclass
class NeuroTokenizerConfig:
    """Configuration for NeuroLM EEG Tokenizer"""
    # EEG processing parameters
    sampling_rate: int = 200  # Hz
    window_size: int = 200    # samples (1 second at 200Hz)
    n_channels: int = 8       # OpenBCI channels
    
    # Vector quantization parameters
    n_embed: int = 8192       # Codebook size
    embed_dim: int = 128      # Embedding dimension
    decay: float = 0.99       # EMA decay
    
    # Neural transformer parameters
    n_layer: int = 6          # Transformer layers
    n_head: int = 8           # Attention heads
    n_embd: int = 512         # Hidden dimension
    dropout: float = 0.1
    bias: bool = False
    
    # Real-time processing
    overlap_ratio: float = 0.5  # Window overlap
    block_size: int = 1024      # Maximum sequence length


class TemporalConv(nn.Module):
    """EEG to Patch Embedding using temporal convolutions"""
    
    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()
        
        # Project to transformer dimension
        self.projection = nn.Sequential(
            nn.Linear(400, 512),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: EEG data [B, N_channels, T_samples]
        Returns:
            Embedded features [B, N_channels, hidden_dim]
        """
        B, N, T = x.shape
        x = x.unsqueeze(1)  # [B, 1, N, T]
        
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        
        x = rearrange(x, 'B C N T -> B N (T C)')
        x = self.projection(x)
        
        return x


class NormEMAVectorQuantizer(nn.Module):
    """Normalized EMA Vector Quantizer from NeuroLM"""
    
    def __init__(self, n_embed, embedding_dim, beta=1.0, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_embed = n_embed
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        
        # Initialize embeddings
        self.embedding = nn.Embedding(n_embed, embedding_dim)
        self.embedding.weight.data.uniform_(-1/n_embed, 1/n_embed)
        
        # EMA parameters
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())

    def forward(self, z):
        """
        Args:
            z: Input features [B, T, D]
        Returns:
            quantized: Quantized features [B, T, D]
            loss: VQ loss
            indices: Quantization indices [B, T]
        """
        B, T, D = z.shape
        z_flattened = z.view(-1, D)
        
        # L2 normalize
        z_flattened = F.normalize(z_flattened, p=2, dim=1)
        embed_normalized = F.normalize(self.embedding.weight, p=2, dim=1)
        
        # Compute distances
        distances = torch.cdist(z_flattened, embed_normalized)
        
        # Find closest embeddings
        indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(indices, embed_normalized)
        
        # Compute VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), z_flattened)
        q_latent_loss = F.mse_loss(quantized, z_flattened.detach())
        loss = q_latent_loss + self.beta * e_latent_loss
        
        # Straight-through estimator
        quantized = z_flattened + (quantized - z_flattened).detach()
        
        # Update EMA
        if self.training:
            self._update_ema(z_flattened, indices)
        
        quantized = quantized.view(B, T, D)
        indices = indices.view(B, T)
        
        return quantized, loss, indices
    
    def _update_ema(self, z, indices):
        """Update EMA statistics"""
        indices_onehot = F.one_hot(indices, self.n_embed).float()
        cluster_size = indices_onehot.sum(0)
        embed_sum = torch.matmul(indices_onehot.t(), z)
        
        self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        
        # Update embeddings
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)


class NeuroLMTokenizer(nn.Module):
    """
    NeuroLM-based EEG Tokenizer for real-time attention metrics
    Converts raw EEG signals into discrete neural tokens
    """
    
    def __init__(self, config: NeuroTokenizerConfig):
        super().__init__()
        self.config = config
        
        # Temporal feature extraction
        self.patch_embed = TemporalConv(in_chans=1, out_chans=16)
        
        # Positional and channel embeddings
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)
        self.chan_embed = nn.Embedding(len(STANDARD_1020), config.n_embd)
        
        # Vector quantizer
        self.quantizer = NormEMAVectorQuantizer(
            n_embed=config.n_embed,
            embedding_dim=config.embed_dim,
            decay=config.decay
        )
        
        # Projection layers
        self.pre_quant_proj = nn.Linear(config.n_embd, config.embed_dim)
        self.post_quant_proj = nn.Linear(config.embed_dim, config.n_embd)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_channel_indices(self, channel_names):
        """Convert channel names to indices"""
        indices = []
        for name in channel_names:
            try:
                indices.append(STANDARD_1020.index(name.upper()))
            except ValueError:
                indices.append(STANDARD_1020.index('pad'))  # Unknown channels
        return indices
    
    def preprocess_eeg(self, eeg_data, channel_names):
        """
        Preprocess raw EEG data for tokenization
        
        Args:
            eeg_data: Raw EEG [n_channels, n_samples]
            channel_names: List of channel names
            
        Returns:
            Preprocessed EEG ready for tokenization
        """
        # Convert to tensor
        if not isinstance(eeg_data, torch.Tensor):
            eeg_data = torch.FloatTensor(eeg_data)
        
        # Normalize to microvolts and standardize
        eeg_data = eeg_data / 100.0  # Assuming input in raw units
        
        # Segment into windows
        n_channels, n_samples = eeg_data.shape
        window_size = self.config.window_size
        overlap = int(window_size * self.config.overlap_ratio)
        step = window_size - overlap
        
        windows = []
        for i in range(0, n_samples - window_size + 1, step):
            window = eeg_data[:, i:i + window_size]
            windows.append(window)
        
        if not windows:
            # If signal is too short, pad it
            padded = torch.zeros(n_channels, window_size)
            padded[:, :n_samples] = eeg_data
            windows = [padded]
        
        # Stack windows
        windowed_data = torch.stack(windows)  # [n_windows, n_channels, window_size]
        
        # Get channel indices
        channel_indices = self.get_channel_indices(channel_names)
        
        return windowed_data, channel_indices
    
    def forward(self, eeg_data, channel_names, return_tokens_only=False):
        """
        Tokenize EEG data
        
        Args:
            eeg_data: Raw EEG [n_channels, n_samples] or preprocessed [B, n_channels, window_size]
            channel_names: List of channel names
            return_tokens_only: If True, return only token indices
            
        Returns:
            tokens: Discrete neural tokens
            features: Continuous features (if return_tokens_only=False)
            loss: VQ loss (if training)
        """
        # Preprocess if needed
        if eeg_data.dim() == 2:
            eeg_data, channel_indices = self.preprocess_eeg(eeg_data, channel_names)
        else:
            channel_indices = self.get_channel_indices(channel_names)
        
        B, N, T = eeg_data.shape
        
        # Extract temporal features
        features = self.patch_embed(eeg_data)  # [B, N, hidden_dim]
        
        # Add positional and channel embeddings
        pos_ids = torch.arange(N, device=features.device).unsqueeze(0).expand(B, -1)
        chan_ids = torch.tensor(channel_indices[:N], device=features.device).unsqueeze(0).expand(B, -1)
        
        pos_emb = self.pos_embed(pos_ids)
        chan_emb = self.chan_embed(chan_ids)
        
        features = features + pos_emb + chan_emb
        features = self.dropout(features)
        
        # Project to quantization space
        quant_input = self.pre_quant_proj(features)
        
        # Vector quantization
        quantized, vq_loss, tokens = self.quantizer(quant_input)
        
        if return_tokens_only:
            return tokens
        
        # Project back to feature space
        quantized_features = self.post_quant_proj(quantized)
        
        return {
            'tokens': tokens,
            'features': quantized_features,
            'continuous_features': features,
            'vq_loss': vq_loss
        }
    
    def encode_to_tokens(self, eeg_data, channel_names):
        """
        Encode EEG data to discrete tokens for real-time processing
        
        Args:
            eeg_data: Raw EEG [n_channels, n_samples]
            channel_names: List of channel names
            
        Returns:
            tokens: Discrete neural tokens [n_windows, n_channels]
        """
        self.eval()
        with torch.no_grad():
            tokens = self.forward(eeg_data, channel_names, return_tokens_only=True)
        return tokens
    
    def save_tokenizer(self, path):
        """Save tokenizer state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)
    
    @classmethod
    def load_tokenizer(cls, path, device='cpu'):
        """Load tokenizer from saved state"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        tokenizer = cls(config)
        tokenizer.load_state_dict(checkpoint['model_state_dict'])
        return tokenizer


class RealTimeEEGTokenizer:
    """
    Real-time wrapper for NeuroLM tokenizer
    Handles streaming EEG data and produces continuous token streams
    """
    
    def __init__(self, tokenizer: NeuroLMTokenizer, buffer_size: int = 2048):
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.config = tokenizer.config
        
        # Initialize buffers
        self.reset_buffers()
    
    def reset_buffers(self):
        """Reset internal buffers"""
        self.eeg_buffer = torch.zeros(self.config.n_channels, self.buffer_size)
        self.buffer_ptr = 0
        self.token_history = []
    
    def add_eeg_sample(self, sample):
        """
        Add new EEG sample to buffer
        
        Args:
            sample: EEG sample [n_channels] or [1, n_channels]
        """
        if isinstance(sample, np.ndarray):
            sample = torch.FloatTensor(sample)
        
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)
        
        # Add to circular buffer
        self.eeg_buffer[:, self.buffer_ptr] = sample.squeeze()
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_size
    
    def get_latest_tokens(self, channel_names, n_windows=1):
        """
        Get tokens from latest EEG data
        
        Args:
            channel_names: List of channel names
            n_windows: Number of recent windows to tokenize
            
        Returns:
            tokens: Latest neural tokens
        """
        window_size = self.config.window_size
        total_samples = n_windows * window_size
        
        if self.buffer_ptr < total_samples:
            # Not enough data yet
            return None
        
        # Extract recent data
        start_idx = (self.buffer_ptr - total_samples) % self.buffer_size
        if start_idx + total_samples <= self.buffer_size:
            recent_data = self.eeg_buffer[:, start_idx:start_idx + total_samples]
        else:
            # Handle wrap-around
            part1 = self.eeg_buffer[:, start_idx:]
            part2 = self.eeg_buffer[:, :total_samples - (self.buffer_size - start_idx)]
            recent_data = torch.cat([part1, part2], dim=1)
        
        # Tokenize
        tokens = self.tokenizer.encode_to_tokens(recent_data, channel_names)
        
        # Store in history
        self.token_history.append(tokens)
        if len(self.token_history) > 100:  # Keep last 100 token sets
            self.token_history.pop(0)
        
        return tokens
    
    def get_token_sequence(self, sequence_length=10):
        """
        Get sequence of recent tokens for temporal analysis
        
        Args:
            sequence_length: Number of recent token sets to return
            
        Returns:
            Token sequence for temporal modeling
        """
        if len(self.token_history) < sequence_length:
            return None
        
        recent_tokens = self.token_history[-sequence_length:]
        return torch.stack(recent_tokens)


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = NeuroTokenizerConfig(
        sampling_rate=250,
        n_channels=8,
        n_embed=1024,
        embed_dim=64
    )
    
    # Create tokenizer
    tokenizer = NeuroLMTokenizer(config)
    
    # Test with synthetic EEG data
    synthetic_eeg = torch.randn(8, 1000)  # 8 channels, 1000 samples (4 seconds at 250Hz)
    channel_names = ['FP1', 'FP2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    
    # Tokenize
    result = tokenizer(synthetic_eeg, channel_names)
    print(f"Tokens shape: {result['tokens'].shape}")
    print(f"Features shape: {result['features'].shape}")
    print(f"VQ Loss: {result['vq_loss'].item():.4f}")
    
    # Test real-time tokenizer
    rt_tokenizer = RealTimeEEGTokenizer(tokenizer)
    
    # Simulate streaming data
    for i in range(500):
        sample = torch.randn(8)
        rt_tokenizer.add_eeg_sample(sample)
        
        if i > 200 and i % 50 == 0:  # After some warmup
            tokens = rt_tokenizer.get_latest_tokens(channel_names)
            if tokens is not None:
                print(f"Step {i}: Latest tokens shape: {tokens.shape}")
