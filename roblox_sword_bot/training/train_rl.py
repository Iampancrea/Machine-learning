"""
Reinforcement Learning Trainer
Uses Stable Baselines3 SAC for continuous-action training with the custom RobloxGymEnv.
SAC is off-policy and sample-efficient — critical when each env step costs real game time.
"""
import torch
import torch.nn as nn
import os
import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback

from models.network import SpatialCNN
from utils.config import load_config
from training.env import RobloxGymEnv


class RobloxFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for SB3 that matches our HybridNetwork architecture.
    Extracts structured features (1D) and spatial features (5-channel CNN).
    """
    def __init__(self, observation_space, cnn_output_dim=32, cnn_channels=5):
        struct_dim = observation_space.spaces['structured'].shape[0]
        features_dim = struct_dim + cnn_output_dim
        super(RobloxFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Instantiate SpatialCNN with 5 input channels (4 grayscale + 1 enemy mask)
        self.cnn = SpatialCNN(output_dim=cnn_output_dim, in_channels=cnn_channels)
        
    def forward(self, observations) -> torch.Tensor:
        struct = observations['structured']
        cnn = observations['cnn_frame']
        
        cnn_features = self.cnn(cnn)
        return torch.cat([struct, cnn_features], dim=1)


def find_latest_checkpoint(save_dir: Path) -> Optional[Path]:
    """Helper to find the latest saved RL checkpoint"""
    if not save_dir.exists():
        return None
    
    final_path = save_dir / "sac_roblox_final.zip"
    if final_path.exists():
        return final_path
        
    checkpoints = list(save_dir.glob("sac_roblox_*_steps.zip"))
    if not checkpoints:
        return None
        
    def get_step_count(path: Path):
        try:
            return int(path.stem.split('_')[2])
        except (IndexError, ValueError):
            return 0
            
    checkpoints.sort(key=get_step_count, reverse=True)
    return checkpoints[0]


class RLTrainer:
    def __init__(self, config: dict):
        self.config = config

    def train(self, episodes: int = 1000):
        """Main RL training loop using SAC"""
        print("\n⚔️ Starting Phase 2: Reinforcement Learning (SAC)")
        
        rl_config = self.config.get('reinforcement_learning', {})
        save_dir = Path("checkpoints/rl_models")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Gym Environment (no BC checkpoint needed for SAC)
        env = RobloxGymEnv(config=self.config)
        
        # CNN channels: 4 grayscale frames + 1 enemy mask = 5
        cnn_frame_stack = self.config.get('features', {}).get('cnn_frame_stack', 4)
        cnn_channels = cnn_frame_stack + 1
        
        # Policy kwargs for SAC's actor-critic architecture
        hidden_layers = self.config.get('model', {}).get('hidden_layers', [128, 64])
        policy_kwargs = dict(
            features_extractor_class=RobloxFeatureExtractor,
            features_extractor_kwargs=dict(
                cnn_output_dim=self.config.get('model', {}).get('cnn_output_dim', 32),
                cnn_channels=cnn_channels
            ),
            net_arch=dict(pi=hidden_layers, qf=hidden_layers)  # Actor and Q-function branches
        )
        
        device = self.config.get('hardware', {}).get('device', 'cpu')
        
        # Check if we can resume training
        latest_checkpoint = find_latest_checkpoint(save_dir)
        
        if latest_checkpoint:
            print(f"\n♻️ Found existing training checkpoint: {latest_checkpoint}")
            print("🧠 Loading SAC weights to resume training...")
            model = SAC.load(
                str(latest_checkpoint),
                env=env,
                device=device,
                custom_objects={
                    "learning_rate": rl_config.get('learning_rate', 0.0003),
                    "buffer_size": rl_config.get('buffer_size', 10000),
                    "policy_kwargs": policy_kwargs
                }
            )
        else:
            print("\n🧠 Initializing SAC Agent from scratch...")
            model = SAC(
                "MultiInputPolicy",
                env,
                learning_rate=rl_config.get('learning_rate', 0.0003),
                buffer_size=rl_config.get('buffer_size', 50000),
                learning_starts=rl_config.get('learning_starts', 500),
                batch_size=rl_config.get('batch_size', 64),
                tau=rl_config.get('tau', 0.005),
                gamma=rl_config.get('gamma', 0.99),
                ent_coef='auto',  # SAC auto-tunes entropy for maximum exploration
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=device
            )
        
        # Setup saving
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=str(save_dir),
            name_prefix="sac_roblox"
        )
        
        print("\n🔥 Training Started! (Press ESC at any time to hard-kill the process)")
        total_timesteps = episodes * rl_config.get('steps_per_episode', 500)
        
        try:
            model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        except KeyboardInterrupt:
            print("\n⏹️ Training manually stopped.")
        
        # Save final model
        final_path = save_dir / "sac_roblox_final.zip"
        model.save(str(final_path))
        print(f"✅ Final model saved to: {final_path}")
        
        env.close()

if __name__ == "__main__":
    from utils.config import load_config
    cfg = load_config("configs/default_config.yaml")
    trainer = RLTrainer(config=cfg.config)
    trainer.train()
