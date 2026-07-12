"""
Behavior Cloning Trainer
Trains model to mimic human gameplay from recorded data
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

from models.network import MLPNetwork, create_model
from utils.config import Config


class GameplayDataset(Dataset):
    """Dataset for loading recorded gameplay data"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.sessions = list(self.data_dir.glob("*.npz"))
        
        if len(self.sessions) == 0:
            raise ValueError(f"No training data found in {data_dir}")
        
        print(f"Found {len(self.sessions)} recording sessions")
        
        # Load all data
        self.features_list = []
        self.actions_list = []
        
        for session_file in self.sessions:
            data = np.load(session_file, allow_pickle=True)
            
            features = data['features']
            actions_json = data['actions']
            
            # Parse actions from JSON
            actions = [json.dumps(a) if isinstance(a, dict) else a for a in actions_json]
            
            self.features_list.append(features)
            self.actions_list.extend(actions)
        
        # Concatenate all features
        self.features = np.concatenate(self.features_list, axis=0)
        
        # Parse actions into tensors
        self.actions = self._parse_actions()
        
        print(f"Loaded {len(self.features)} total samples")
        print(f"Feature shape: {self.features.shape}")
        print(f"Action classes: {len(self.action_mapping)}")
    
    def _parse_actions(self) -> torch.Tensor:
        """Parse action strings into class labels"""
        # Create action mapping
        all_actions = set()
        parsed_actions = []
        
        for action_json in self.actions_list:
            try:
                if isinstance(action_json, str):
                    action = json.loads(action_json)
                else:
                    action = action_json
                
                # Convert action to string key
                keys = sorted(action.get('keys', []))
                click = 1 if action.get('click', False) else 0
                mouse_dx = 1 if action.get('mouse_dx', 0) > 0.1 else (-1 if action.get('mouse_dx', 0) < -0.1 else 0)
                mouse_dy = 1 if action.get('mouse_dy', 0) > 0.1 else (-1 if action.get('mouse_dy', 0) < -0.1 else 0)
                
                action_key = f"{','.join(keys)}_{click}_{mouse_dx}_{mouse_dy}"
                all_actions.add(action_key)
                parsed_actions.append(action_key)
            except:
                parsed_actions.append("none_0_0_0")
        
        # Create mapping
        self.action_mapping = {a: i for i, a in enumerate(sorted(all_actions))}
        self.reverse_mapping = {i: a for a, i in self.action_mapping.items()}
        
        # Convert to tensor
        action_tensor = torch.tensor([self.action_mapping[a] for a in parsed_actions], dtype=torch.long)
        
        return action_tensor
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            self.actions[idx]
        )


class BehaviorCloningTrainer:
    """Train model using behavior cloning (supervised learning)"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('hardware', {}).get('device', 'cpu')
        
        # Initialize model
        self.feature_dim = config.get('features', {}).get('feature_history_length', 10) * 15  # Approximate
        self.num_actions = 8  # Will be updated when dataset is loaded
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, epochs: int = 50, batch_size: int = 32):
        """Train the behavior cloning model"""
        
        # Load dataset
        data_dir = self.config.get('paths', {}).get('data_dir', './data') + '/recordings'
        
        try:
            dataset = GameplayDataset(data_dir)
        except FileNotFoundError:
            print(f"\n⚠️  No training data found in {data_dir}")
            print("Please run 'python main.py record' first to collect gameplay data")
            return
        
        self.num_actions = len(dataset.action_mapping)
        
        # Create model
        self.model = create_model(
            model_type='mlp',
            input_dim=dataset.features.shape[1],
            output_dim=self.num_actions,
            config=self.config.get('model', {})
        ).to(self.device)
        
        # Create data loader
        val_split = self.config.get('behavior_cloning', {}).get('validation_split', 0.2)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
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
        print(f"\nTraining on {self.device}")
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print(f"Number of action classes: {self.num_actions}\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('behavior_cloning', {}).get('early_stopping_patience', 10)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for features, actions in pbar:
                features = features.to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, actions)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += actions.size(0)
                train_correct += (predicted == actions).sum().item()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{train_correct/train_total:.3f}"
                })
            
            train_loss /= train_size
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, actions in val_loader:
                    features = features.to(self.device)
                    actions = actions.to(self.device)
                    
                    outputs = self.model(features)
                    loss = self.criterion(outputs, actions)
                    
                    val_loss += loss.item() * features.size(0)
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
                self.save_model("best_model.pth")
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
                self.save_model(f"checkpoint_epoch_{epoch+1}.pth")
        
        print("\n✅ Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def save_model(self, filename: str):
        """Save model to file"""
        model_dir = Path(self.config.get('paths', {}).get('model_dir', './checkpoints'))
        model_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_actions': self.num_actions,
            'config': self.config
        }, save_path)
        
        print(f"Model saved to: {save_path}")
    
    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = create_model(
            model_type='mlp',
            input_dim=self.feature_dim,
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
