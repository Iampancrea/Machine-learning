"""
Reinforcement Learning Trainer
Uses Stable Baselines3 for PPO training with the custom RobloxGymEnv.
Pre-loads weights from Behavior Cloning (BC) to jumpstart training.
"""
import torch
import torch.nn as nn
import os
import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback

from models.network import SpatialCNN
from utils.config import load_config
from training.env import RobloxGymEnv


class RobloxFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for SB3 that matches our HybridNetwork architecture.
    Extracts structured features (1D) and spatial features (2D CNN).
    """
    def __init__(self, observation_space, cnn_output_dim=32):
        struct_dim = observation_space.spaces['structured'].shape[0]
        features_dim = struct_dim + cnn_output_dim
        super(RobloxFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Instantiate our exact SpatialCNN used in BC
        self.cnn = SpatialCNN(output_dim=cnn_output_dim)
        
    def forward(self, observations) -> torch.Tensor:
        struct = observations['structured']
        cnn = observations['cnn_frame']
        
        cnn_features = self.cnn(cnn)
        return torch.cat([struct, cnn_features], dim=1)


def transfer_bc_weights(ppo_model, bc_checkpoint_path: str):
    """
    Surgically injects BC weights into the SB3 PPO model graph.
    """
    print(f"\n💉 Injecting BC weights from {bc_checkpoint_path}...")
    bc_checkpoint = torch.load(bc_checkpoint_path, map_location='cpu')
    bc_state = bc_checkpoint['model_state_dict']
    
    policy_state = ppo_model.policy.state_dict()
    
    # 1. Transfer CNN feature extractor weights
    cnn_transferred = 0
    for k, v in bc_state.items():
        if k.startswith('cnn.'):
            # In SB3, our feature extractor is under features_extractor.cnn
            target_key = f"features_extractor.{k}"
            if target_key in policy_state:
                target_v = policy_state[target_key]
                if target_v.shape == v.shape:
                    policy_state[target_key] = v
                    cnn_transferred += 1
                elif k == 'cnn.conv_stack.0.weight' and target_v.shape[1] == 2 and v.shape[1] == 1:
                    print("🔪 Surgically adapting Conv1 weights from 1-channel to 2-channel...")
                    new_w = target_v.clone()
                    new_w[:, 0:1, :, :] = v   # Grayscale channel = BC weights
                    new_w[:, 1:2, :, :] = 0.0 # Mask channel = initialized to 0
                    policy_state[target_key] = new_w
                    cnn_transferred += 1
    
    # 2. Transfer Actor MLP weights (BC's mlp_head → SB3's policy_net + action_net)
    # BC HybridNetwork: mlp_head = Linear→ReLU→Dropout→Linear→ReLU→Dropout→Linear
    # SB3 with net_arch=dict(pi=[128, 64]):
    #   mlp_extractor.policy_net.0 ↔ mlp_head.0  (features → 128)
    #   mlp_extractor.policy_net.2 ↔ mlp_head.3  (128 → 64)
    #   action_net                 ↔ mlp_head.6  (64 → num_actions)
    # NOTE: HybridNetwork uses 'mlp_head', not 'fusion_head'!
    # The mlp_head is: Linear→ReLU→Dropout→Linear→ReLU→Dropout→Linear
    # Indices: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear, 4=ReLU, 5=Dropout, 6=Linear
    mlp_map = {
        'mlp_head.0.weight': 'mlp_extractor.policy_net.0.weight',
        'mlp_head.0.bias': 'mlp_extractor.policy_net.0.bias',
        'mlp_head.3.weight': 'mlp_extractor.policy_net.2.weight',
        'mlp_head.3.bias': 'mlp_extractor.policy_net.2.bias',
        'mlp_head.6.weight': 'action_net.weight',
        'mlp_head.6.bias': 'action_net.bias'
    }
    
    mlp_transferred = 0
    for bc_k, sb3_k in mlp_map.items():
        if bc_k in bc_state and sb3_k in policy_state:
            v_bc = bc_state[bc_k]
            v_sb3 = policy_state[sb3_k]
            
            if v_bc.shape == v_sb3.shape:
                policy_state[sb3_k] = v_bc
                mlp_transferred += 1
            elif bc_k == 'mlp_head.0.weight' and v_bc.shape[1] < v_sb3.shape[1]:
                print("🔪 Surgically adapting MLP weights for new Player State features...")
                new_w = v_sb3.clone()
                # Old base features (0-10) -> (0-10)
                new_w[:, 0:11] = v_bc[:, 0:11]
                # New features (11-12) -> initialized to 0
                new_w[:, 11:13] = 0.0
                # Remaining historical + CNN features shifted by 2
                new_w[:, 13:] = v_bc[:, 11:]
                
                policy_state[sb3_k] = new_w
                mlp_transferred += 1
                
    ppo_model.policy.load_state_dict(policy_state)
    print(f"✅ Transferred {cnn_transferred} CNN layers and {mlp_transferred} Actor MLP layers.")


def find_latest_checkpoint(save_dir: Path) -> Optional[Path]:
    """Helper to find the latest saved RL checkpoint"""
    if not save_dir.exists():
        return None
    
    final_path = save_dir / "ppo_roblox_final.zip"
    if final_path.exists():
        return final_path
        
    checkpoints = list(save_dir.glob("ppo_roblox_*_steps.zip"))
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
        """Main RL training loop"""
        print("\n⚔️ Starting Phase 2: Reinforcement Learning (PPO)")
        
        bc_checkpoint = "checkpoints/best_model.pth"
        rl_config = self.config.get('reinforcement_learning', {})
        save_dir = Path("checkpoints/rl_models")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Gym Environment
        env = RobloxGymEnv(config=self.config, checkpoint_path=bc_checkpoint)
        
        # Policy kwargs to match HybridNetwork (128, 64 hidden layers)
        hidden_layers = self.config.get('model', {}).get('hidden_layers', [128, 64])
        policy_kwargs = dict(
            features_extractor_class=RobloxFeatureExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=self.config.get('model', {}).get('cnn_output_dim', 32)),
            net_arch=dict(pi=hidden_layers, vf=hidden_layers) # Actor and Critic branches
        )
        
        device = self.config.get('hardware', {}).get('device', 'cpu')
        
        # Check if we can resume training
        latest_checkpoint = find_latest_checkpoint(save_dir)
        
        if latest_checkpoint:
            print(f"\n♻️ Found existing training checkpoint: {latest_checkpoint}")
            print("🧠 Loading PPO weights to resume training...")
            model = PPO.load(
                str(latest_checkpoint),
                env=env,
                device=device,
                custom_objects={
                    "learning_rate": rl_config.get('learning_rate', 0.00003),
                    "n_steps": rl_config.get('steps_per_episode', 256),
                    "policy_kwargs": policy_kwargs
                }
            )
        else:
            print("\n🧠 Initializing PPO Agent from scratch...")
            model = PPO(
                "MultiInputPolicy",
                env,
                learning_rate=rl_config.get('learning_rate', 0.0003),
                n_steps=rl_config.get('steps_per_episode', 500),
                batch_size=64,
                n_epochs=10,
                gamma=rl_config.get('gamma', 0.99),
                ent_coef=0.05, # HIGH ENTROPY for forced curiosity (spamming clicks)
                target_kl=0.03, # CLAMP KL DIVERGENCE to protect BC weights
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=device
            )
            # Inject BC weights so it knows how to walk
            transfer_bc_weights(model, bc_checkpoint)
        
        # Setup saving
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=str(save_dir),
            name_prefix="ppo_roblox"
        )
        
        print("\n🔥 Training Started! (Press ESC at any time to hard-kill the process)")
        total_timesteps = episodes * rl_config.get('steps_per_episode', 500)
        
        try:
            model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        except KeyboardInterrupt:
            print("\n⏹️ Training manually stopped.")
        
        # Save final model
        final_path = save_dir / "ppo_roblox_final.zip"
        model.save(str(final_path))
        print(f"✅ Final model saved to: {final_path}")
        
        env.close()

if __name__ == "__main__":
    from utils.config import load_config
    cfg = load_config("configs/default_config.yaml")
    trainer = RLTrainer(config=cfg.config)
    trainer.train()
