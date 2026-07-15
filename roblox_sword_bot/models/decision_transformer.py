"""
Decision Transformer architecture for offline RL.
This model receives a sequence of (Return-To-Go, State, Action) tuples and outputs
the predicted action for the current step using causal self-attention.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class TrajectoryEmbedder(nn.Module):
    """Projects states, actions, and returns into a shared latent space"""
    def __init__(self, struct_dim: int, cnn_channels: int, action_dim: int, embed_dim: int):
        super().__init__()
        
        # CNN for visual states (matches HybridNetwork spatial processing)
        self.cnn = nn.Sequential(
            nn.Conv2d(cnn_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 10, 256),
            nn.ReLU()
        )
        
        # Projectors for each modality
        self.state_proj = nn.Linear(struct_dim + 256, embed_dim)
        self.action_proj = nn.Linear(action_dim, embed_dim)
        self.return_proj = nn.Linear(1, embed_dim)
        
    def forward(self, struct: torch.Tensor, cnn: torch.Tensor, action: torch.Tensor, ret: torch.Tensor):
        # B x T x ...
        B, T = struct.shape[:2]
        
        # Process CNN (flatten B and T)
        cnn_flat = cnn.view(B * T, *cnn.shape[2:])
        cnn_feat = self.cnn(cnn_flat)
        cnn_feat = cnn_feat.view(B, T, -1)
        
        # Process Struct
        state_feat = torch.cat([struct, cnn_feat], dim=-1)
        state_embed = self.state_proj(state_feat)
        
        action_embed = self.action_proj(action)
        return_embed = self.return_proj(ret.unsqueeze(-1))
        
        return state_embed, action_embed, return_embed

class DecisionTransformer(nn.Module):
    def __init__(
        self,
        struct_dim: int = 15,
        cnn_channels: int = 5,
        action_dim: int = 9,  # 5 keys + 2 clicks + 2 mouse
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_length: int = 64
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.action_dim = action_dim
        
        # Modality embedders
        self.embedder = TrajectoryEmbedder(struct_dim, cnn_channels, action_dim, embed_dim)
        
        # Positional Encoding (Time embeddings)
        self.pos_emb = nn.Embedding(max_length, embed_dim)
        
        # Core Transformer (GPT style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Action Head Prediction (predicts continuous logit/value for each action dim)
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, action_dim)
        )
        
    def forward(self, struct: torch.Tensor, cnn: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor):
        """
        Input shapes:
            struct:  (B, T, struct_dim)
            cnn:     (B, T, C, H, W)
            actions: (B, T, action_dim)
            returns: (B, T)
        """
        B, T = struct.shape[:2]
        assert T <= self.max_length, f"Sequence length {T} exceeds max {self.max_length}"
        
        # 1. Embed Modalities
        s_emb, a_emb, r_emb = self.embedder(struct, cnn, actions, returns)
        
        # 2. Add Positional Embedding (1 to T)
        time_steps = torch.arange(T, device=struct.device).unsqueeze(0).repeat(B, 1)
        p_emb = self.pos_emb(time_steps)
        
        s_emb = s_emb + p_emb
        a_emb = a_emb + p_emb
        r_emb = r_emb + p_emb
        
        # 3. Interleave Sequence
        # Sequence format: (R_1, S_1, A_1, R_2, S_2, A_2, ... R_T, S_T, A_T)
        # We stack and reshape to interleave them along the sequence dimension
        stacked = torch.stack([r_emb, s_emb, a_emb], dim=2) # (B, T, 3, E)
        seq = stacked.view(B, T * 3, self.embed_dim)        # (B, 3T, E)
        
        # 4. Causal Mask
        # We need a mask to prevent looking into the future
        # Mask shape: (3T, 3T). True means *masked* (not allowed to attend) in PyTorch.
        # Actually PyTorch Transformer expects mask to be (3T, 3T) with -inf for masked, 0.0 for unmasked,
        # OR boolean True for masked. We'll use the boolean True/False approach.
        # generate_square_subsequent_mask returns a mask where True is masked out.
        mask = nn.Transformer.generate_square_subsequent_mask(T * 3, device=seq.device)
        
        # 5. Transformer Forward
        out_seq = self.transformer(seq, mask=mask, is_causal=True) # (B, 3T, E)
        
        # 6. Action Prediction
        # We want to predict A_t based on (R_t, S_t). 
        # The representation after S_t is at index 1, 4, 7... (which is 1 + 3*t)
        # Sequence indices for states are 1, 4, 7...
        state_idx = torch.arange(1, T * 3, 3, device=seq.device)
        state_representations = out_seq[:, state_idx, :] # (B, T, E)
        
        # Predict actions
        action_preds = self.action_head(state_representations) # (B, T, action_dim)
        
        return action_preds
        
    def get_action(self, struct_seq, cnn_seq, action_seq, return_seq, deterministic=True):
        """
        Convenience method for the live inference controller.
        Inputs are unbatched sequences: (T, ...)
        """
        self.eval()
        with torch.no_grad():
            s = struct_seq.unsqueeze(0)
            c = cnn_seq.unsqueeze(0)
            a = action_seq.unsqueeze(0)
            r = return_seq.unsqueeze(0)
            
            preds = self.forward(s, c, a, r) # (1, T, action_dim)
            
            # Get the prediction for the final step in the sequence
            last_pred = preds[0, -1, :]
            
            # Decode the action_dim vector into the specific heads (Keys, Clicks, Mouse)
            # Keys (0-4), Clicks (5-6), Mouse (7-8)
            keys = last_pred[0:5]
            clicks = last_pred[5:7]
            mouse = last_pred[7:9]
            
            # Apply activations exactly like HybridNetwork
            key_probs = torch.sigmoid(keys)
            click_probs = torch.sigmoid(clicks)
            mouse_out = torch.tanh(mouse)
            
            if deterministic:
                key_out = (key_probs > 0.5).int().cpu().numpy()
                click_out = (click_probs > 0.5).int().cpu().numpy()
            else:
                key_out = torch.bernoulli(key_probs).int().cpu().numpy()
                click_out = torch.bernoulli(click_probs).int().cpu().numpy()
                
            mouse_out = mouse_out.cpu().numpy()
            
            # Build dictionary (mapping indices exactly as in HybridNetwork)
            key_map = ['W', 'A', 'S', 'D', 'Space']
            pressed_keys = [k for i, k in enumerate(key_map) if key_out[i] == 1]
            
            action_dict = {
                'keys': pressed_keys,
                'click_left': bool(click_out[0]),
                'click_right': bool(click_out[1]),
                'mouse_dx': float(mouse_out[0]),
                'mouse_dy': float(mouse_out[1])
            }
            
            return action_dict
