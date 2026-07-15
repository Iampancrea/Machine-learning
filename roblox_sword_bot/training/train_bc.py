"""
Behavior Cloning Trainer
Trains the hybrid model (structured features + CNN) to mimic human gameplay.

Updated to load BOTH structured features AND raw 80x60 grayscale CNN frames
from .npz recording files. Supports training on cloud GPUs (Kaggle, SageMaker,
Lightning AI) — just upload the .npz files as a dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from models.network import MLPNetwork, HybridNetwork, create_model
from utils.config import Config


class GameplayDataset(Dataset):
    """Dataset for loading recorded gameplay data (structured + CNN frames)"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.sessions = list(self.data_dir.glob("*.npz"))
        
        if len(self.sessions) == 0:
            raise ValueError(f"No training data found in {data_dir}")
        
        print(f"Found {len(self.sessions)} recording sessions")
        
        # Load all data
        self.features_list = []
        self.cnn_frames_list = []
        self.actions_list = []
        
        for session_file in self.sessions:
            data = np.load(session_file, allow_pickle=True)
            
            features = data['features']
            actions_json = data['actions']
            
            # Load CNN frames (80x60 grayscale, float32)
            if 'cnn_frames' in data:
                cnn_frames = data['cnn_frames']
                self.cnn_frames_list.append(cnn_frames)
            else:
                # Fallback for old recordings without CNN frames
                print(f"  ⚠️  {session_file.name}: no cnn_frames — generating dummy zeros")
                dummy_frames = np.zeros((len(features), 60, 80), dtype=np.float32)
                self.cnn_frames_list.append(dummy_frames)
            
            # Parse actions from JSON
            actions = [json.dumps(a) if isinstance(a, dict) else a for a in actions_json]
        self._load_data()
        
        # Parse actions into tensors
        self.actions_keys, self.actions_clicks, self.actions_mouse = self._parse_actions()
        
        print(f"Loaded {len(self.features)} total samples")
        print(f"Structured feature shape: {self.features.shape}")
        print(f"CNN frame shape: {self.cnn_frames.shape}")
        
    def _load_data(self):
        # Check for HDF5 dataset first (Cloud optimization)
        h5_path = self.data_dir.parent / 'dataset.h5'
        if h5_path.exists():
            print(f"Loading optimized HDF5 dataset from {h5_path}")
            import h5py
            # For simplicity in this demo, load into RAM. True streaming requires keeping h5f open.
            with h5py.File(h5_path, 'r') as h5f:
                self.features = np.array(h5f['features'])
                self.cnn_frames = np.array(h5f['cnn_frames'])
                self.actions_list = [a.decode('utf-8') if isinstance(a, bytes) else a for a in h5f['actions']]
            return

        # Fallback to NPZ directory loading
        npz_files = list(self.data_dir.glob('*.npz'))
        if not npz_files:
            raise FileNotFoundError(f"No recording files found in {self.data_dir}")
            
        print(f"Loading {len(npz_files)} recording files...")
        for f in npz_files:
            try:
                with np.load(f, allow_pickle=True) as data:
                    self.features_list.append(data['features'])
                    self.cnn_frames_list.append(data['cnn_frames'])
                    
                    if 'actions' in data:
                        actions = data['actions']
                        if len(actions) > 0 and isinstance(actions[0], dict):
                            self.actions_list.extend([json.dumps(a) for a in actions])
                        else:
                            self.actions_list.extend(actions)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        if not self.features_list:
            raise ValueError("No valid data loaded")
            
        self.features = np.concatenate(self.features_list, axis=0)
        self.cnn_frames = np.concatenate(self.cnn_frames_list, axis=0)

    def _parse_actions(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse action strings into multi-hot and continuous tensors"""
        keys_list, clicks_list, mouse_list = [], [], []
        valid_keys = ["w", "a", "s", "d", "space"]
        
        for action_json in self.actions_list:
            try:
                if isinstance(action_json, str):
                    action = json.loads(action_json)
                else:
                    action = action_json
                
                raw_keys = [str(k).lower() for k in action.get('keys', [])]
                key_vec = [1.0 if k in raw_keys else 0.0 for k in valid_keys]
                
                click_left = 1.0 if action.get('click_left', action.get('click', False)) else 0.0
                click_right = 1.0 if action.get('click_right', False) else 0.0
                
                mouse_dx = float(action.get('mouse_dx', 0.0))
                mouse_dy = float(action.get('mouse_dy', 0.0))
                
                keys_list.append(key_vec)
                clicks_list.append([click_left, click_right])
                mouse_list.append([mouse_dx, mouse_dy])
            except Exception as e:
                print(f"⚠️ Warning: Failed to parse action {action_json}. Error: {e}")
                keys_list.append([0.0] * 5)
                clicks_list.append([0.0] * 2)
                mouse_list.append([0.0] * 2)
        
        self.action_mapping = {"continuous": True} # dummy for compatibility
        
        return (torch.tensor(keys_list, dtype=torch.float32),
                torch.tensor(clicks_list, dtype=torch.float32),
                torch.tensor(mouse_list, dtype=torch.float32))
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Structured features
        struct_feat = torch.FloatTensor(self.features[idx])
        
        # CNN frame padding
        frame_data = self.cnn_frames[idx]
        if len(frame_data.shape) == 2:  # Legacy (60, 80)
            cnn_frame = torch.FloatTensor(frame_data).unsqueeze(0) / 255.0
            cnn_frame = cnn_frame.repeat(5, 1, 1) # pad to 5 channels
        else:
            cnn_frame = torch.FloatTensor(frame_data) / 255.0
            if cnn_frame.shape[0] < 5:
                pad = torch.zeros((5 - cnn_frame.shape[0], 60, 80))
                cnn_frame = torch.cat([cnn_frame, pad], dim=0)
        
        return struct_feat, cnn_frame, self.actions_keys[idx], self.actions_clicks[idx], self.actions_mouse[idx]


class BehaviorCloningTrainer:
    """Train hybrid model using behavior cloning (supervised learning)"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('hardware', {}).get('device', 'cpu')
        
        self.model = None
        self.optimizer = None
        self.num_actions = 9  # 5 keys + 2 clicks + 2 mouse
        
    def train(self, epochs: int = 50, batch_size: int = 32):
        """Train the hybrid behavior cloning model"""
        
        # Load dataset
        data_dir = self.config.get('paths', {}).get('data_dir', './data') + '/recordings'
        
        try:
            dataset = GameplayDataset(data_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n⚠️  {e}")
            print("Please run 'python main.py record' first to collect gameplay data")
            return
        
        self.num_actions = len(dataset.action_mapping)
        structured_dim = dataset.features.shape[1]
        
        # Create hybrid model
        model_type = self.config.get('model', {}).get('type', 'hybrid')
        
        if model_type == 'hybrid':
            self.model = HybridNetwork(
                structured_dim=structured_dim,
                cnn_output_dim=self.config.get('model', {}).get('cnn_output_dim', 32),
                hidden_layers=self.config.get('model', {}).get('hidden_layers', [128, 64]),
                output_dim=self.num_actions,
                dropout=self.config.get('model', {}).get('dropout', 0.1)
            ).to(self.device)
        else:
            # Fallback to MLP-only (ignores CNN frames)
            self.model = create_model(
                model_type='mlp',
                input_dim=structured_dim,
                output_dim=self.num_actions,
                config=self.config.get('model', {})
            ).to(self.device)
        
        # Create data loaders (chronological split to prevent temporal leakage)
        val_split = self.config.get('behavior_cloning', {}).get('validation_split', 0.2)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        from torch.utils.data import Subset
        train_dataset = Subset(dataset, range(0, train_size))
        val_dataset = Subset(dataset, range(train_size, len(dataset)))
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )
        
        # Setup optimizer
        lr = self.config.get('behavior_cloning', {}).get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        is_hybrid = isinstance(self.model, HybridNetwork)
        
        print(f"\nTraining on {self.device}")
        print(f"Model type: {'hybrid (structured + CNN)' if is_hybrid else 'MLP only'}")
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print(f"Number of action classes: {self.num_actions}\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('behavior_cloning', {}).get('early_stopping_patience', 10)
        
        for epoch in range(epochs):
            # ─── Training phase ───
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for struct_feats, cnn_frames, tgt_keys, tgt_clicks, tgt_mouse in pbar:
                struct_feats = struct_feats.to(self.device)
                cnn_frames = cnn_frames.to(self.device)
                tgt_keys = tgt_keys.to(self.device)
                tgt_clicks = tgt_clicks.to(self.device)
                tgt_mouse = tgt_mouse.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if is_hybrid:
                    out_keys, out_clicks, out_mouse = self.model(struct_feats, cnn_frames)
                else:
                    continue # MLP fallback not supported with continuous heads
                
                bce = nn.BCEWithLogitsLoss()
                mse = nn.MSELoss()
                
                loss_keys = bce(out_keys, tgt_keys)
                loss_clicks = bce(out_clicks, tgt_clicks)
                loss_mouse = mse(out_mouse, tgt_mouse)
                
                loss = loss_keys + loss_clicks + loss_mouse
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item() * struct_feats.size(0)
                train_total += struct_feats.size(0)
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'mse': f"{loss_mouse.item():.4f}"
                })
            
            train_loss /= train_size
            train_acc = train_correct / train_total
            
            # ─── Validation phase ───
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for struct_feats, cnn_frames, tgt_keys, tgt_clicks, tgt_mouse in val_loader:
                    struct_feats = struct_feats.to(self.device)
                    cnn_frames = cnn_frames.to(self.device)
                    tgt_keys = tgt_keys.to(self.device)
                    tgt_clicks = tgt_clicks.to(self.device)
                    tgt_mouse = tgt_mouse.to(self.device)
                    
                    if is_hybrid:
                        out_keys, out_clicks, out_mouse = self.model(struct_feats, cnn_frames)
                    else:
                        continue
                    
                    bce = nn.BCEWithLogitsLoss()
                    mse = nn.MSELoss()
                    
                    loss = bce(out_keys, tgt_keys) + bce(out_clicks, tgt_clicks) + mse(out_mouse, tgt_mouse)
                    
                    val_loss += loss.item() * struct_feats.size(0)
                    val_total += struct_feats.size(0)
            
            val_loss /= val_size
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model.pth", dataset.action_mapping)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every N epochs
            checkpoint_freq = self.config.get('logging', {}).get('checkpoint_frequency', 5)
            if (epoch + 1) % checkpoint_freq == 0:
                self.save_model(f"checkpoint_epoch_{epoch+1}.pth", dataset.action_mapping)
        
        print("\n✅ Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def save_model(self, filename: str, action_mapping: dict = None):
        """Save model to file"""
        model_dir = Path(self.config.get('paths', {}).get('model_dir', './checkpoints'))
        model_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = model_dir / filename
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_actions': self.num_actions,
            'model_type': 'hybrid' if isinstance(self.model, HybridNetwork) else 'mlp',
            'config': self.config,
        }
        
        if action_mapping:
            save_dict['action_mapping'] = action_mapping
        
        # Save structured_dim for hybrid model reconstruction
        if isinstance(self.model, HybridNetwork):
            save_dict['structured_dim'] = self.model.structured_dim
        
        torch.save(save_dict, save_path)
        print(f"Model saved to: {save_path}")
    
    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        
        model_type = checkpoint.get('model_type', 'mlp')
        
        if model_type == 'hybrid':
            structured_dim = checkpoint.get('structured_dim', 38)
            self.model = HybridNetwork(
                structured_dim=structured_dim,
                cnn_output_dim=self.config.get('model', {}).get('cnn_output_dim', 32),
                hidden_layers=self.config.get('model', {}).get('hidden_layers', [128, 64]),
                output_dim=checkpoint['num_actions'],
                dropout=self.config.get('model', {}).get('dropout', 0.1)
            ).to(self.device)
        else:
            self.model = create_model(
                model_type='mlp',
                input_dim=checkpoint.get('structured_dim', 150),
                output_dim=checkpoint['num_actions'],
                config=self.config.get('model', {})
            ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.num_actions = checkpoint['num_actions']
        
        print(f"Model loaded from: {path}")


if __name__ == "__main__":
    # Test trainer initialization
    from utils.config import load_config
    config = load_config()
    trainer = BehaviorCloningTrainer(config=config.config)
    print("Behavior Cloning Trainer initialized")
    print("Run 'python main.py train_bc' to start training")
