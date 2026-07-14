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
        
        # Observation space: Dict with structured (1D) and cnn_frame (2D/3D)
        self.observation_space = spaces.Dict({
            "structured": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.feature_engineer.get_structured_dim(),), 
                dtype=np.float32
            ),
            "cnn_frame": spaces.Box(
                low=0.0, high=1.0,
                shape=(2, cnn_res[1], cnn_res[0]),
                dtype=np.float32
            )
        })
        
        # Reward configuration
        self.reward_kill = rewards_cfg.get('kill', 10.0)
        self.reward_death = rewards_cfg.get('death', -10.0)
        self.reward_safe_zone = rewards_cfg.get('safe_zone_penalty', -0.05)
        self.reward_safe_zone_leave = rewards_cfg.get('safe_zone_leave', 2.0)
        self.reward_safe_zone_reenter = rewards_cfg.get('safe_zone_reenter', -2.0)
        self.reward_health_drop = rewards_cfg.get('health_drop_multiplier', -1.0)
        self.reward_trigger_bonus = rewards_cfg.get('trigger_bonus', 0.05)
        self.reward_trigger_penalty = rewards_cfg.get('trigger_penalty', 0.0)
        self.reward_idle = rewards_cfg.get('idle_penalty', -0.01)
        
        # OCR scan interval
        self.ocr_scan_interval = config.get('features', {}).get(
            'kill_log', {}).get('scan_interval_frames', 15)
            
        # Action repeat
        self.action_repeat = config.get('actions', {}).get('action_repeat', 4)
        
        # State tracking
        self.is_dead = False
        self.kill_cooldown = 0
        self.step_count = 0
        self.episode_kills = 0
        self.episode_deaths = 0
        self.episode_reward = 0.0
        self.last_action = None
        self.show_vision = config.get("debug_vision", True)
        
        # Dense Reward: Safe Zone state machine
        # has_left_safe_zone = False → bot just spawned/respawned, no penalty for being in safe zone
        # has_left_safe_zone = True  → bot has been out in the field, re-entering = cowardice penalty
        self.has_left_safe_zone = False
        self.prev_in_safe_zone = True  # assume we start in safe zone
        self.prev_health = 1.0
        
    def _decode_action(self, action_idx: int) -> dict:
        """Convert integer action from PPO to our standard input dictionary"""
        action_str = self.reverse_action_mapping.get(action_idx, "none_0_0_0_0")
        parts = action_str.split('_')
        
        if len(parts) >= 5:
            keys = parts[0].split(',') if parts[0] and parts[0] != 'none' else []
            # Strip out '1' or 'KEY.1' to prevent unequipping the sword during fight
            keys = [k for k in keys if k not in ['1', 'KEY.1']]
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
        
        # Wait for respawn: if death is detected, wait until it is no longer detected
        if self.game_detector.detect_death():
            print("⏳ Player is dead/respawning. Waiting for respawn...")
            # Sleep to let Roblox handle the respawn animation
            time.sleep(2.0)
            # Wait until health bar is healthy again (template match is low)
            start_wait = time.time()
            while self.game_detector.detect_death():
                time.sleep(0.5)
                # Failsafe: don't loop forever if Roblox crashed or something
                if time.time() - start_wait > 12.0:
                    print("⚠️ Respawn wait timed out (12s). Resuming anyway.")
                    break
            print("✅ Respawn detected. Equipping sword...")
            time.sleep(1.0) # Wait for character to settle
            
        # Ensure sword is equipped on spawn/respawn
        self.input_controller.press_key('1', duration=0.15)
        
        self.is_dead = False
        self.kill_cooldown = 0
        self.step_count = 0
        self.episode_kills = 0
        self.episode_deaths = 0
        self.episode_reward = 0.0
        self.last_action = None
        self.last_step_time = time.time()
        self.has_left_safe_zone = False
        self.prev_in_safe_zone = True
        self.prev_health = 1.0
        
        print(f"\n--- NEW EPISODE ---")
        
        # Get initial observation
        frame = self.screen_capture.capture()
        enemies = self.game_detector.detect_enemies(frame)
        in_safe_zone = self.game_detector.detect_safe_zone(frame)
        player_health = self.game_detector.get_player_health()
        struct, cnn = self.feature_engineer.extract_features(
            frame=frame, 
            enemies=enemies, 
            in_safe_zone=in_safe_zone,
            player_health=player_health,
            last_action=None
        )
        # cnn is now shape (2, H, W)
        cnn = cnn.astype(np.float32) / 255.0
        
        obs = {
            "structured": struct.astype(np.float32),
            "cnn_frame": cnn
        }
        
        return obs, {}

    def step(self, action_idx):
        action_dict = self._decode_action(int(action_idx))
        self.last_action = action_dict
        
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for _ in range(self.action_repeat):
            self.step_count += 1
            
            # Enforce 30 FPS limit to prevent PC from overheating
            target_frame_time = 1.0 / 30.0
            elapsed = time.time() - self.last_step_time
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
            self.last_step_time = time.time()
            
            # 1. Execute Action
            self.input_controller.execute_action(action_dict)
            
            # 2. Get New State (only basics needed during repeat)
            frame = self.screen_capture.capture()
            enemies = self.game_detector.detect_enemies(frame)
            in_safe_zone = self.game_detector.detect_safe_zone(frame)
            player_health = self.game_detector.get_player_health()
            
            # 3. Reward Shaping
            step_reward = 0.0
            
            # ── Pain Penalty (Health Drop) ──────────────────────────────
            if player_health < self.prev_health and not self.is_dead:
                damage_taken = self.prev_health - player_health
                pain_penalty = damage_taken * self.reward_health_drop
                step_reward += pain_penalty
                print(f"  🩸 TOOK DAMAGE! {damage_taken*100:.0f}%. Penalty: {pain_penalty:.2f}")
            self.prev_health = player_health
            
            # ── Trigger Discipline (Clicking at enemies) ─────────────────
            if action_dict.get('click_left', False) and not self.is_dead:
                swung_at_enemy = False
                for enemy in enemies:
                    if 'player_center' in enemy:
                        px, py = enemy['player_center']
                        # Distance from center crosshair (400, 300)
                        dist = ((px - 400)**2 + (py - 300)**2)**0.5
                        if dist < 150.0:  # Enemy is in front of us
                            swung_at_enemy = True
                            break
                if swung_at_enemy:
                    step_reward += self.reward_trigger_bonus
                else:
                    step_reward += self.reward_trigger_penalty
            
            # ── Death detection (EVERY frame — fast, no OCR) ─────────────
            if not self.is_dead and self.game_detector.detect_death():
                step_reward += self.reward_death
                terminated = True
                self.is_dead = True
                self.episode_deaths += 1
                # Reset safe zone flag on death so respawn doesn't get penalized
                self.has_left_safe_zone = False
                self.prev_in_safe_zone = True
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
                    step_reward += self.reward_kill
                    self.kill_cooldown = 3  # 3 OCR checks = ~1.5 seconds cooldown
                    self.episode_kills += 1
                    victim = kill_log_status.get('victim', 'unknown')
                    print(f"\n🩸 KILL! You killed: {victim} | "
                          f"Reward: +{self.reward_kill} | Step {self.step_count} | "
                          f"Total kills: {self.episode_kills}")
                    
                # Check OCR for death as backup
                if kill_log_status['death'] and not self.is_dead:
                    step_reward += self.reward_death
                    terminated = True
                    self.is_dead = True
                    self.episode_deaths += 1
                    killer = kill_log_status.get('killer', 'unknown')
                    print(f"\n☠️  KILLED BY (OCR): {killer} | "
                          f"Reward: {self.reward_death} | Step {self.step_count}")
            
            # ── Continuous rewards (every frame) ─────────────────────────
            if not terminated:
                # ── Dense Reward: Safe Zone State Machine ─────────────
                # Transition: was in safe zone → now outside = LEFT the zone
                if self.prev_in_safe_zone and not in_safe_zone:
                    if not self.has_left_safe_zone:
                        # First time leaving after spawn/respawn → big reward
                        step_reward += self.reward_safe_zone_leave
                        self.has_left_safe_zone = True
                        print(f"  🏃 LEFT SAFE ZONE! +{self.reward_safe_zone_leave} reward")
                
                # Transition: was outside → now back in safe zone = COWARDICE
                if not self.prev_in_safe_zone and in_safe_zone:
                    if self.has_left_safe_zone:
                        # Re-entered safe zone without dying → penalty
                        step_reward += self.reward_safe_zone_reenter
                        print(f"  🐔 RE-ENTERED SAFE ZONE! {self.reward_safe_zone_reenter} penalty")
                
                self.prev_in_safe_zone = in_safe_zone
                
                # Per-frame safe zone camping drip penalty (on top of the -2 event)
                if in_safe_zone and self.has_left_safe_zone:
                    step_reward += self.reward_safe_zone
                
                # Idle penalty (no keys pressed and no clicks)
                if action_dict and not action_dict.get('keys') and not action_dict.get('click_left'):
                    step_reward += self.reward_idle
            
            total_reward += step_reward
            self.episode_reward += step_reward
            
            # Periodic status update
            if self.step_count % 60 == 0:
                print(f"  [Step {self.step_count}] Enemies: {len(enemies)} | "
                      f"Safe: {in_safe_zone} | Reward: {self.episode_reward:.2f} | "
                      f"Kills: {self.episode_kills}")
                      
            if getattr(self, 'show_vision', False):
                import cv2
                debug_frame = frame.copy()
                for enemy in enemies:
                    px, py = enemy['player_center']
                    cv2.circle(debug_frame, (px, py), 5, (0, 255, 0), -1)
                    if isinstance(enemy['hp_bar'], (tuple, list, tuple)) and len(enemy['hp_bar']) == 4:
                        x, y, w, h = enemy['hp_bar']
                        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(debug_frame, "Enemy", (px + 10, py), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Shrink it so it doesn't cover the whole screen (1/3rd size)
                dh, dw = debug_frame.shape[:2]
                debug_frame = cv2.resize(debug_frame, (dw // 3, dh // 3))
                cv2.putText(debug_frame, "AI VISION FEED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                bgr_frame = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Roblox Bot Vision", bgr_frame)
                cv2.waitKey(1)

            if terminated:
                break
                
        # Only compute the CNN features on the FINAL frame of the action repeat loop to save CPU
        player_health = self.game_detector.get_player_health()
        struct, cnn = self.feature_engineer.extract_features(
            frame=frame, 
            enemies=enemies, 
            in_safe_zone=in_safe_zone,
            player_health=player_health,
            last_action=self.last_action
        )
        cnn = cnn.astype(np.float32) / 255.0
        obs = {
            "structured": struct.astype(np.float32),
            "cnn_frame": cnn
        }
        
        return obs, total_reward, terminated, truncated, {}
