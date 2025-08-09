"""
NeuroLM-based Attention Metrics Model
Foundation model for extracting attention and engagement metrics from EEG neural tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
from transformers import GPT2Config, GPT2Model
from neurolm_tokenizer import NeuroLMTokenizer, NeuroTokenizerConfig


@dataclass
class AttentionModelConfig:
    """Configuration for NeuroLM Attention Model"""
    # Model architecture
    vocab_size: int = 8192        # Same as tokenizer codebook size
    n_layer: int = 12             # Transformer layers
    n_head: int = 12              # Attention heads
    n_embd: int = 768             # Hidden dimension
    dropout: float = 0.1
    
    # Attention-specific parameters
    n_attention_classes: int = 3   # Low, Medium, High attention
    n_engagement_classes: int = 3  # Low, Medium, High engagement
    temporal_context: int = 16     # Number of time steps to consider
    
    # Multi-task learning
    use_auxiliary_tasks: bool = True
    auxiliary_weight: float = 0.3
    
    # Real-time processing
    max_sequence_length: int = 512
    prediction_horizon: int = 1    # Predict next N steps


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional causal masking"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None, causal=False):
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply masks
        if causal:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            attn = attn.masked_fill(~causal_mask, float('-inf'))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        
        return out, attn


class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward"""
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x, mask=None, causal=False):
        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(self.ln1(x), mask=mask, causal=causal)
        x = x + attn_out
        
        # Feedforward with residual connection
        x = x + self.mlp(self.ln2(x))
        
        return x, attn_weights


class NeuroLMAttentionModel(nn.Module):
    """
    NeuroLM-based model for real-time attention and engagement prediction
    Uses neural tokens from EEG data to predict attention states
    """
    
    def __init__(self, config: AttentionModelConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.max_sequence_length, config.n_embd)
        
        # Transformer backbone
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Task-specific heads
        self.attention_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd // 2, config.n_attention_classes)
        )
        
        self.engagement_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd // 2, config.n_engagement_classes)
        )
        
        # Regression heads for continuous metrics
        self.alpha_theta_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 1)
        )
        
        self.beta_alpha_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 1)
        )
        
        self.workload_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 1)
        )
        
        # Auxiliary tasks (if enabled)
        if config.use_auxiliary_tasks:
            self.next_token_head = nn.Linear(config.n_embd, config.vocab_size)
            self.channel_prediction_head = nn.Linear(config.n_embd, 8)  # Predict missing channels
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, token_ids, attention_mask=None, labels=None):
        """
        Forward pass
        
        Args:
            token_ids: Neural token IDs [B, T]
            attention_mask: Attention mask [B, T]
            labels: Ground truth labels (dict with keys: attention, engagement, etc.)
            
        Returns:
            Dictionary with predictions and losses
        """
        B, T = token_ids.shape
        device = token_ids.device
        
        # Embeddings
        token_emb = self.token_embed(token_ids)
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(pos_ids)
        
        x = self.dropout(token_emb + pos_emb)
        
        # Transformer layers
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn = block(x, mask=attention_mask, causal=True)
            attention_weights.append(attn)
        
        x = self.ln_f(x)
        
        # Use last token for classification (or mean pooling)
        if attention_mask is not None:
            # Use last non-masked token
            last_token_idx = attention_mask.sum(dim=1) - 1
            pooled = x[torch.arange(B), last_token_idx]
        else:
            pooled = x[:, -1]  # Last token
        
        # Task predictions
        attention_logits = self.attention_head(pooled)
        engagement_logits = self.engagement_head(pooled)
        
        # Continuous metrics
        alpha_theta_ratio = self.alpha_theta_head(pooled).squeeze(-1)
        beta_alpha_ratio = self.beta_alpha_head(pooled).squeeze(-1)
        workload = self.workload_head(pooled).squeeze(-1)
        
        outputs = {
            'attention_logits': attention_logits,
            'engagement_logits': engagement_logits,
            'alpha_theta_ratio': alpha_theta_ratio,
            'beta_alpha_ratio': beta_alpha_ratio,
            'workload': workload,
            'hidden_states': x,
            'attention_weights': attention_weights
        }
        
        # Auxiliary tasks
        if self.config.use_auxiliary_tasks:
            next_token_logits = self.next_token_head(x)
            channel_pred = self.channel_prediction_head(pooled)
            outputs.update({
                'next_token_logits': next_token_logits,
                'channel_predictions': channel_pred
            })
        
        # Compute losses if labels provided
        if labels is not None:
            losses = self._compute_losses(outputs, labels)
            outputs['losses'] = losses
            outputs['total_loss'] = sum(losses.values())
        
        return outputs
    
    def _compute_losses(self, outputs, labels):
        """Compute task-specific losses"""
        losses = {}
        
        # Classification losses
        if 'attention' in labels:
            losses['attention'] = F.cross_entropy(
                outputs['attention_logits'], labels['attention']
            )
        
        if 'engagement' in labels:
            losses['engagement'] = F.cross_entropy(
                outputs['engagement_logits'], labels['engagement']
            )
        
        # Regression losses
        if 'alpha_theta_ratio' in labels:
            losses['alpha_theta'] = F.mse_loss(
                outputs['alpha_theta_ratio'], labels['alpha_theta_ratio']
            )
        
        if 'beta_alpha_ratio' in labels:
            losses['beta_alpha'] = F.mse_loss(
                outputs['beta_alpha_ratio'], labels['beta_alpha_ratio']
            )
        
        if 'workload' in labels:
            losses['workload'] = F.mse_loss(
                outputs['workload'], labels['workload']
            )
        
        # Auxiliary losses
        if self.config.use_auxiliary_tasks:
            if 'next_tokens' in labels:
                losses['next_token'] = F.cross_entropy(
                    outputs['next_token_logits'].view(-1, self.config.vocab_size),
                    labels['next_tokens'].view(-1)
                ) * self.config.auxiliary_weight
        
        return losses
    
    def predict_attention_state(self, token_ids, attention_mask=None):
        """
        Predict attention state from neural tokens
        
        Args:
            token_ids: Neural token IDs [B, T]
            attention_mask: Attention mask [B, T]
            
        Returns:
            Dictionary with attention predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(token_ids, attention_mask)
            
            # Convert logits to probabilities and predictions
            attention_probs = F.softmax(outputs['attention_logits'], dim=-1)
            engagement_probs = F.softmax(outputs['engagement_logits'], dim=-1)
            
            attention_pred = torch.argmax(attention_probs, dim=-1)
            engagement_pred = torch.argmax(engagement_probs, dim=-1)
            
            return {
                'attention_state': attention_pred,
                'attention_confidence': torch.max(attention_probs, dim=-1)[0],
                'engagement_state': engagement_pred,
                'engagement_confidence': torch.max(engagement_probs, dim=-1)[0],
                'alpha_theta_ratio': outputs['alpha_theta_ratio'],
                'beta_alpha_ratio': outputs['beta_alpha_ratio'],
                'workload': outputs['workload'],
                'attention_probs': attention_probs,
                'engagement_probs': engagement_probs
            }
    
    def get_attention_embeddings(self, token_ids, attention_mask=None):
        """
        Extract attention-relevant embeddings for downstream analysis
        
        Args:
            token_ids: Neural token IDs [B, T]
            attention_mask: Attention mask [B, T]
            
        Returns:
            Attention embeddings [B, hidden_dim]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(token_ids, attention_mask)
            
            # Use last token representation
            B, T = token_ids.shape
            if attention_mask is not None:
                last_token_idx = attention_mask.sum(dim=1) - 1
                embeddings = outputs['hidden_states'][torch.arange(B), last_token_idx]
            else:
                embeddings = outputs['hidden_states'][:, -1]
            
            return embeddings


class NeuroLMAttentionTrainer:
    """Trainer for NeuroLM attention model"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(
            token_ids=batch['token_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch.get('labels')
        )
        
        loss = outputs['total_loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'individual_losses': {k: v.item() for k, v in outputs['losses'].items()}
        }
    
    def evaluate(self, eval_dataloader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        correct_attention = 0
        correct_engagement = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.model(
                    token_ids=batch['token_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels')
                )
                
                total_loss += outputs['total_loss'].item()
                
                # Accuracy calculation
                if 'labels' in batch and 'attention' in batch['labels']:
                    attention_pred = torch.argmax(outputs['attention_logits'], dim=-1)
                    correct_attention += (attention_pred == batch['labels']['attention']).sum().item()
                
                if 'labels' in batch and 'engagement' in batch['labels']:
                    engagement_pred = torch.argmax(outputs['engagement_logits'], dim=-1)
                    correct_engagement += (engagement_pred == batch['labels']['engagement']).sum().item()
                
                total_samples += batch['token_ids'].size(0)
        
        return {
            'avg_loss': total_loss / len(eval_dataloader),
            'attention_accuracy': correct_attention / total_samples if total_samples > 0 else 0,
            'engagement_accuracy': correct_engagement / total_samples if total_samples > 0 else 0
        }
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def load_model(cls, path, tokenizer, device='cpu'):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = NeuroLMAttentionModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        trainer = cls(model, tokenizer, config)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return trainer


# Example usage
if __name__ == "__main__":
    # Test configuration
    config = AttentionModelConfig(
        vocab_size=1024,
        n_layer=6,
        n_head=8,
        n_embd=512
    )
    
    # Create model
    model = NeuroLMAttentionModel(config)
    
    # Test with synthetic tokens
    batch_size = 4
    seq_length = 32
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    outputs = model(token_ids)
    print(f"Attention logits shape: {outputs['attention_logits'].shape}")
    print(f"Engagement logits shape: {outputs['engagement_logits'].shape}")
    print(f"Alpha/theta ratio shape: {outputs['alpha_theta_ratio'].shape}")
    
    # Test prediction
    predictions = model.predict_attention_state(token_ids)
    print(f"Attention states: {predictions['attention_state']}")
    print(f"Attention confidence: {predictions['attention_confidence']}")
    print(f"Alpha/theta ratios: {predictions['alpha_theta_ratio']}")
