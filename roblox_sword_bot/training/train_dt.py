"""
Decision Transformer Training Loop
Trains the model to predict actions from a sequence of (Return-To-Go, State, Action).
Requires HDF5 dataset annotated with returns.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from models.decision_transformer import DecisionTransformer

class TrajectoryDataset(Dataset):
    def __init__(self, h5_path: str, context_len: int = 64):
        self.context_len = context_len
        self.h5_path = h5_path
        
        # We load it all into RAM for this implementation, but for huge datasets,
        # we would keep the file open and stream it.
        print(f"Loading {h5_path} into memory...")
        with h5py.File(h5_path, 'r') as h5f:
            self.features = np.array(h5f['features'], dtype=np.float32)
            self.cnn_frames = np.array(h5f['cnn_frames'], dtype=np.float32) / 255.0
            self.rtg = np.array(h5f['rtg'], dtype=np.float32)
            actions_raw = [a.decode('utf-8') if isinstance(a, bytes) else a for a in h5f['actions']]
            
        self.total_frames = len(self.features)
        
        # Parse actions into a unified float tensor matching the DecisionTransformer action_dim
        # Keys(5), Clicks(2), Mouse(2)
        print("Parsing action strings into tensors...")
        parsed_actions = np.zeros((self.total_frames, 9), dtype=np.float32)
        key_map = {'W': 0, 'A': 1, 'S': 2, 'D': 3, 'Space': 4}
        
        for i, a_str in enumerate(tqdm(actions_raw)):
            try:
                act = json.loads(a_str) if isinstance(a_str, str) and a_str.startswith('{') else {}
                # Keys
                for k in act.get('keys', []):
                    if k in key_map:
                        parsed_actions[i, key_map[k]] = 1.0
                # Clicks
                parsed_actions[i, 5] = 1.0 if act.get('click_left', False) else 0.0
                parsed_actions[i, 6] = 1.0 if act.get('click_right', False) else 0.0
                # Mouse
                parsed_actions[i, 7] = float(act.get('mouse_dx', 0.0))
                parsed_actions[i, 8] = float(act.get('mouse_dy', 0.0))
            except: pass
            
        self.actions = parsed_actions
        print("Dataset ready.")

    def __len__(self):
        # We can start a trajectory anywhere up to total_frames - context_len
        return max(1, self.total_frames - self.context_len)

    def __getitem__(self, idx):
        # Grab a chunk of length context_len
        end_idx = idx + self.context_len
        
        s = self.features[idx:end_idx]
        c = self.cnn_frames[idx:end_idx]
        a = self.actions[idx:end_idx]
        r = self.rtg[idx:end_idx]
        
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(c),
            torch.FloatTensor(a),
            torch.FloatTensor(r)
        )

def train_dt(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training Decision Transformer on {device}")
    
    dataset = TrajectoryDataset(args.data, context_len=args.context_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    # Initialize Model
    model = DecisionTransformer(
        struct_dim=15, 
        cnn_channels=dataset.cnn_frames.shape[1], # usually 5
        action_dim=9,
        max_length=args.context_len
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    
    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for s, c, a, r in pbar:
            s, c, a, r = s.to(device), c.to(device), a.to(device), r.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: predict actions based on the context
            # Note: The model predicts action a_t based on s_t, r_t, and ALL PAST history.
            preds = model(s, c, a, r) # (B, T, action_dim)
            
            # We want to compare the predicted action at step t with the ACTUAL action taken at step t.
            # a is (B, T, action_dim)
            # Keys/Clicks use BCE, Mouse uses MSE
            
            pred_keys_clicks = preds[:, :, 0:7]
            true_keys_clicks = a[:, :, 0:7]
            loss_discrete = bce_loss(pred_keys_clicks, true_keys_clicks)
            
            pred_mouse = torch.tanh(preds[:, :, 7:9]) # apply tanh like bot_controller
            true_mouse = a[:, :, 7:9]
            loss_continuous = mse_loss(pred_mouse, true_mouse)
            
            # Combine losses
            loss = loss_discrete + (loss_continuous * 10.0) # weight mouse heavily for aiming
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} average loss: {total_loss / len(loader):.4f}")
        
        # Save checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/dt_epoch_{epoch+1}.pth")
        
    print("✅ Decision Transformer Training Complete!")
    torch.save(model.state_dict(), "checkpoints/dt_best.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dataset_rtg.h5')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--context_len', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    
    train_dt(args)
