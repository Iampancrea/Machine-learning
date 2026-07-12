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
            if target_key in policy_state and policy_state[target_key].shape == v.shape:
                policy_state[target_key] = v
                cnn_transferred += 1
    
    # 2. Transfer Fusion MLP (Actor) weights
    # In BC, it's fusion_head.0, fusion_head.2, fusion_head.4 (action output)
    # In SB3 with net_arch=dict(pi=[128, 64]), it's:
    #   mlp_extractor.policy_net.0 (matches fusion_head.0)
    #   mlp_extractor.policy_net.2 (matches fusion_head.2)
    #   action_net (matches fusion_head.4)
    mlp_map = {
        'fusion_head.0.weight': 'mlp_extractor.policy_net.0.weight',
        'fusion_head.0.bias': 'mlp_extractor.policy_net.0.bias',
        'fusion_head.2.weight': 'mlp_extractor.policy_net.2.weight',
        'fusion_head.2.bias': 'mlp_extractor.policy_net.2.bias',
        'fusion_head.4.weight': 'action_net.weight',
        'fusion_head.4.bias': 'action_net.bias'
    }
    
    mlp_transferred = 0
    for bc_k, sb3_k in mlp_map.items():
        if bc_k in bc_state and sb3_k in policy_state:
            if bc_state[bc_k].shape == policy_state[sb3_k].shape:
                policy_state[sb3_k] = bc_state[bc_k]
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
                custom_objects={"policy_kwargs": policy_kwargs}
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
                clip_range=rl_config.get('clip_range', 0.2),
                ent_coef=0.05, # HIGH ENTROPY for forced curiosity (spamming clicks)
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
