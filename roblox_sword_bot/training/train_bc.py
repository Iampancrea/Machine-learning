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
            
            self.features_list.append(features)
            self.actions_list.extend(actions)
        
        # Concatenate all features and CNN frames
        self.features = np.concatenate(self.features_list, axis=0)
        self.cnn_frames = np.concatenate(self.cnn_frames_list, axis=0)
        
        # Parse actions into tensors
        self.actions = self._parse_actions()
        
        print(f"Loaded {len(self.features)} total samples")
        print(f"Structured feature shape: {self.features.shape}")
        print(f"CNN frame shape: {self.cnn_frames.shape}")
        print(f"Action classes: {len(self.action_mapping)}")
    
    def _parse_actions(self) -> torch.Tensor:
        """Parse action strings into class labels"""
        all_actions = set()
        parsed_actions = []
        
        for action_json in self.actions_list:
            try:
                if isinstance(action_json, str):
                    action = json.loads(action_json)
                else:
                    action = action_json
                
                # Convert action to string key
                raw_keys = action.get('keys', [])
                junk = {'alt', 'e', 'shift', 'tab', 'ALT', 'E', 'SHIFT', 'TAB',
                        'KEY.SHIFT', 'KEY.TAB', 'KEY.ALT', 'KEY.ALT_L', 'KEY.ALT_R',
                        'KEY.F2', 'KEY.CTRL', 'KEY.CTRL_L', 'KEY.ESC'}
                keys = sorted([k for k in raw_keys if k not in junk])
                click_left = 1 if action.get('click_left', action.get('click', False)) else 0
                click_right = 1 if action.get('click_right', False) else 0
                mouse_dx = 1 if action.get('mouse_dx', 0) > 0.005 else (-1 if action.get('mouse_dx', 0) < -0.005 else 0)
                mouse_dy = 1 if action.get('mouse_dy', 0) > 0.005 else (-1 if action.get('mouse_dy', 0) < -0.005 else 0)
                
                action_key = f"{','.join(keys)}_{click_left}_{click_right}_{mouse_dx}_{mouse_dy}"
                all_actions.add(action_key)
                parsed_actions.append(action_key)
            except:
                parsed_actions.append("none_0_0_0_0")
        
        # Create mapping
        self.action_mapping = {a: i for i, a in enumerate(sorted(all_actions))}
        self.reverse_mapping = {i: a for a, i in self.action_mapping.items()}
        
        # Convert to tensor
        action_tensor = torch.tensor([self.action_mapping[a] for a in parsed_actions], dtype=torch.long)
        
        return action_tensor
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Structured features
        struct_feat = torch.FloatTensor(self.features[idx])
        
        # CNN frame: handle legacy 1-channel data and new 2-channel data
        frame_data = self.cnn_frames[idx]
        if len(frame_data.shape) == 2:  # Legacy (60, 80)
            cnn_frame = torch.FloatTensor(frame_data).unsqueeze(0) / 255.0
            # Pad with empty mask channel to match network expecting (2, 60, 80)
            empty_mask = torch.zeros_like(cnn_frame)
            cnn_frame = torch.cat([cnn_frame, empty_mask], dim=0)
        else:  # New (2, 60, 80)
            cnn_frame = torch.FloatTensor(frame_data) / 255.0
        
        # Action label
        action = self.actions[idx]
        
        return struct_feat, cnn_frame, action


class BehaviorCloningTrainer:
    """Train hybrid model using behavior cloning (supervised learning)"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('hardware', {}).get('device', 'cpu')
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.num_actions = 8  # Updated when dataset is loaded
        
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
            num_workers=0  # Keep at 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
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
            for struct_feats, cnn_frames, actions in pbar:
                struct_feats = struct_feats.to(self.device)
                cnn_frames = cnn_frames.to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if is_hybrid:
                    outputs = self.model(struct_feats, cnn_frames)
                else:
                    outputs = self.model(struct_feats)
                
                loss = self.criterion(outputs, actions)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item() * struct_feats.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += actions.size(0)
                train_correct += (predicted == actions).sum().item()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{train_correct/train_total:.3f}"
                })
            
            train_loss /= train_size
            train_acc = train_correct / train_total
            
            # ─── Validation phase ───
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for struct_feats, cnn_frames, actions in val_loader:
                    struct_feats = struct_feats.to(self.device)
                    cnn_frames = cnn_frames.to(self.device)
                    actions = actions.to(self.device)
                    
                    if is_hybrid:
                        outputs = self.model(struct_feats, cnn_frames)
                    else:
                        outputs = self.model(struct_feats)
                    
                    loss = self.criterion(outputs, actions)
                    
                    val_loss += loss.item() * struct_feats.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += actions.size(0)
                    val_correct += (predicted == actions).sum().item()
            
            val_loss /= val_size
            val_acc = val_correct / val_total
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
            
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
