"""
Gymnasium Environment for Roblox Sword Fight Bot
Connects the game state to the SB3 PPO algorithm.

Reward signals:
    +10.0  — Kill detected via OCR kill log
    -10.0  — Death detected via health bar greying OR OCR kill log
    -0.05  — Per-step safe zone camping penalty
    +0.02  — Per-step bonus when enemies are visible
    -0.01  — Per-step idle penalty (no keys pressed)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import torch

from feature_extraction.screen_processor import ScreenCapture
from feature_extraction.color_detector import GameDetector
from feature_extraction.feature_engineer import FeatureEngineer
from utils.input_control import InputController

class RobloxGymEnv(gym.Env):
    """
    Standardized Gymnasium environment for Roblox live RL training.
    """
    def __init__(self, config: dict, checkpoint_path: str):
        super(RobloxGymEnv, self).__init__()
        
        self.config = config
        rewards_cfg = config.get('rewards', {})
        
        # Core components
        self.screen_capture = ScreenCapture(
            resolution=tuple(config.get('capture', {}).get('resolution', [800, 600])),
            fps=config.get('capture', {}).get('fps', 30)
        )
        self.game_detector = GameDetector(config=config)
        
        feature_history = config.get('features', {}).get('feature_history_length', 10)
        cnn_res = tuple(config.get('features', {}).get('cnn_resolution', [80, 60]))
        self.feature_engineer = FeatureEngineer(
            history_length=feature_history,
            cnn_resolution=cnn_res
        )
        self.input_controller = InputController(config=config)
        
        # Load BC model for action mapping
        print(f"Loading action mapping from BC checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.action_mapping = checkpoint.get('action_mapping', {})
        self.reverse_action_mapping = {v: k for k, v in self.action_mapping.items()}
        self.num_actions = len(self.action_mapping)
        
        # Action space: Discrete based on BC mapping
        self.action_space = spaces.Discrete(self.num_actions)
        
        # Observation space: Dict with structured (1D) and cnn_frame (2D)
        self.observation_space = spaces.Dict({
            "structured": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.feature_engineer.get_structured_dim(),), 
                dtype=np.float32
            ),
            "cnn_frame": spaces.Box(
                low=0.0, high=1.0,
                shape=(1, cnn_res[1], cnn_res[0]),
                dtype=np.float32
            )
        })
        
        # Reward configuration
        self.reward_kill = rewards_cfg.get('kill', 10.0)
        self.reward_death = rewards_cfg.get('death', -10.0)
        self.reward_safe_zone = rewards_cfg.get('safe_zone_penalty', -0.05)
        self.reward_near_enemy = rewards_cfg.get('near_enemy_bonus', 0.02)
        self.reward_idle = rewards_cfg.get('idle_penalty', -0.01)
        
        # OCR scan interval
        self.ocr_scan_interval = config.get('features', {}).get(
            'kill_log', {}).get('scan_interval_frames', 15)
        
        # State tracking
        self.is_dead = False
        self.kill_cooldown = 0
        self.step_count = 0
        self.episode_kills = 0
        self.episode_deaths = 0
        self.episode_reward = 0.0
        self.last_action = None
        
    def _decode_action(self, action_idx: int) -> dict:
        """Convert integer action from PPO to our standard input dictionary"""
        action_str = self.reverse_action_mapping.get(action_idx, "none_0_0_0_0")
        parts = action_str.split('_')
        
        if len(parts) >= 5:
            keys = parts[0].split(',') if parts[0] and parts[0] != 'none' else []
            click_left = parts[1] == '1'
            click_right = parts[2] == '1'
            mouse_dx = int(parts[3])
            mouse_dy = int(parts[4])
            
            return {
                'keys': keys,
                'mouse_dx': mouse_dx * 0.1,
                'mouse_dy': mouse_dy * 0.1,
                'click': click_left,
                'click_left': click_left,
                'click_right': click_right
            }
        return {'keys': [], 'mouse_dx': 0, 'mouse_dy': 0, 'click': False}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.feature_engineer.reset()
        self.input_controller.reset()
        
        self.is_dead = False
        self.kill_cooldown = 0
        self.step_count = 0
        self.episode_kills = 0
        self.episode_deaths = 0
        self.episode_reward = 0.0
        self.last_action = None
        self.last_step_time = time.time()
        
        print(f"\n--- NEW EPISODE ---")
        
        # Get initial observation
        frame = self.screen_capture.capture()
        enemies = self.game_detector.detect_enemies(frame)
        in_safe_zone = self.game_detector.detect_safe_zone(frame)
        
        struct, cnn = self.feature_engineer.extract_features(frame, enemies, in_safe_zone)
        
        # Convert cnn from (H, W) to (1, H, W) for Dict space compatibility
        cnn = np.expand_dims(cnn.astype(np.float32) / 255.0, axis=0)
        
        obs = {
            "structured": struct.astype(np.float32),
            "cnn_frame": cnn
        }
        
        return obs, {}

    def step(self, action_idx):
        self.step_count += 1
        
        # Enforce 30 FPS limit to prevent PC from overheating
        target_frame_time = 1.0 / 30.0
        elapsed = time.time() - self.last_step_time
        if elapsed < target_frame_time:
            time.sleep(target_frame_time - elapsed)
        self.last_step_time = time.time()
        
        # 1. Execute Action
        action_dict = self._decode_action(int(action_idx))
        self.last_action = action_dict
        self.input_controller.execute_action(action_dict)
        
        # 2. Get New State
        frame = self.screen_capture.capture()
        enemies = self.game_detector.detect_enemies(frame)
        in_safe_zone = self.game_detector.detect_safe_zone(frame)
        
        # Extract features for NN
        struct, cnn = self.feature_engineer.extract_features(frame, enemies, in_safe_zone)
        cnn = np.expand_dims(cnn.astype(np.float32) / 255.0, axis=0)
        
        obs = {
            "structured": struct.astype(np.float32),
            "cnn_frame": cnn
        }
        
        # 3. Reward Shaping
        reward = 0.0
        terminated = False
        truncated = False
        
        # ── Death detection (EVERY frame — fast, no OCR) ─────────────
        if not self.is_dead and self.game_detector.detect_death():
            reward += self.reward_death
            terminated = True
            self.is_dead = True
            self.episode_deaths += 1
            print(f"\n☠️  DEATH DETECTED (health bar greyed) | "
                  f"Reward: {self.reward_death} | Step {self.step_count} | "
                  f"Episode kills: {self.episode_kills}")
        
        # ── Kill log OCR (every N frames — slower, uses OCR) ─────────
        if self.step_count % self.ocr_scan_interval == 0:
            kill_log_status = self.game_detector.detect_kill_log()
            
            # Check for kill (we killed someone) with cooldown
            if self.kill_cooldown > 0:
                self.kill_cooldown -= 1
                
            if kill_log_status['kill'] and self.kill_cooldown == 0:
                reward += self.reward_kill
                self.kill_cooldown = 3  # 3 OCR checks = ~1.5 seconds cooldown
                self.episode_kills += 1
                victim = kill_log_status.get('victim', 'unknown')
                print(f"\n🩸 KILL! You killed: {victim} | "
                      f"Reward: +{self.reward_kill} | Step {self.step_count} | "
                      f"Total kills: {self.episode_kills}")
                
            # Check OCR for death as backup (in case health bar check missed it)
            if kill_log_status['death'] and not self.is_dead:
                reward += self.reward_death
                terminated = True
                self.is_dead = True
                self.episode_deaths += 1
                killer = kill_log_status.get('killer', 'unknown')
                print(f"\n☠️  KILLED BY (OCR): {killer} | "
                      f"Reward: {self.reward_death} | Step {self.step_count}")
        
        # ── Continuous rewards (every frame) ─────────────────────────
        if not terminated:
            # Safe zone camping penalty
            if in_safe_zone:
                reward += self.reward_safe_zone
            
            # Enemy engagement bonus (enemies visible = we're in the fight)
            if enemies:
                reward += self.reward_near_enemy
            
            # Idle penalty (no keys pressed and no clicks)
            if action_dict and not action_dict.get('keys') and not action_dict.get('click_left'):
                reward += self.reward_idle
        
        self.episode_reward += reward
        
        # Periodic status update
        if self.step_count % 60 == 0:
            print(f"  [Step {self.step_count}] Enemies: {len(enemies)} | "
                  f"Safe: {in_safe_zone} | Reward: {self.episode_reward:.2f} | "
                  f"Kills: {self.episode_kills}")
            
        return obs, reward, terminated, truncated, {}
