"""
Gymnasium Environment for Roblox Sword Fight Bot
Connects the game state to the SB3 SAC algorithm.

ACTION SPACE: Box(7,) continuous
    [0] mouse_dx  — continuous camera turn (left/right)
    [1] mouse_dy  — continuous camera tilt (up/down)
    [2] key_w     — forward (thresholded > 0)
    [3] key_a     — left strafe (thresholded > 0)
    [4] key_s     — backward (thresholded > 0)
    [5] key_d     — right strafe (thresholded > 0)
    [6] click_left — sword swing (thresholded > 0)

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
    Uses SAC-compatible continuous action space.
    """
    def __init__(self, config: dict):
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
        cnn_frame_stack = config.get('features', {}).get('cnn_frame_stack', 4)
        self.feature_engineer = FeatureEngineer(
            history_length=feature_history,
            cnn_resolution=cnn_res,
            cnn_frame_stack=cnn_frame_stack
        )
        self.input_controller = InputController(config=config)
        
        # SAC Continuous Action Space: Box(8,)
        # [mouse_dx, mouse_dy, key_w, key_a, key_s, key_d, key_space, click_left]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Observation space: Dict with structured (1D) and cnn_frame (5-channel)
        cnn_channels = cnn_frame_stack + 1  # 4 grayscale + 1 enemy mask = 5
        self.observation_space = spaces.Dict({
            "structured": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.feature_engineer.get_structured_dim(),), 
                dtype=np.float32
            ),
            "cnn_frame": spaces.Box(
                low=0.0, high=1.0,
                shape=(cnn_channels, cnn_res[1], cnn_res[0]),
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
        
        # Mouse sensitivity scaling for SAC
        self.mouse_scale = config.get('actions', {}).get('mouse_sensitivity', 0.5)
        
        # State tracking
        self.is_dead = False
        self.kill_cooldown = 0
        self.step_count = 0
        self.episode_kills = 0
        self.episode_deaths = 0
        self.episode_reward = 0.0
        self.last_action = None
        self.show_vision = config.get("debug_vision", False)
        
        # Dense Reward: Safe Zone state machine
        # has_left_safe_zone = False → bot just spawned/respawned, no penalty for being in safe zone
        # has_left_safe_zone = True  → bot has been out in the field, re-entering = cowardice penalty
        self.has_left_safe_zone = False
        self.prev_in_safe_zone = True  # assume we start in safe zone
        self.prev_health = 1.0
        
    def _decode_sac_action(self, action_vector: np.ndarray) -> dict:
        """
        Convert SAC's continuous Box(8,) output into our standard input dictionary.
        
        Mouse dx/dy are used directly as continuous values.
        Key/click outputs are thresholded at 0 to produce binary presses.
        """
        mouse_dx = float(action_vector[0]) * self.mouse_scale
        mouse_dy = float(action_vector[1]) * self.mouse_scale * 0.05  # Restrict vertical tilt to keep camera level
        
        # Threshold continuous outputs into binary key presses
        keys = []
        if action_vector[2] > 0: keys.append('W')
        if action_vector[3] > 0: keys.append('A')
        if action_vector[4] > 0: keys.append('S')
        if action_vector[5] > 0: keys.append('D')
        if action_vector[6] > 0: keys.append('SPACE')
        
        click_left = bool(action_vector[7] > 0)
        
        return {
            'keys': keys,
            'mouse_dx': mouse_dx,
            'mouse_dy': mouse_dy,
            'click': click_left,
            'click_left': click_left,
            'click_right': False
        }

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
            time.sleep(2.0) # Wait longer for character and inventory to load
            
        # Ensure sword is equipped on spawn/respawn
        self.input_controller.press_key('1', duration=0.2)
        time.sleep(0.1)
        self.input_controller.press_key('1', duration=0.2)
        
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
        # cnn is now shape (5, H, W) — 4 grayscale + 1 enemy mask
        cnn = cnn.astype(np.float32) / 255.0
        
        obs = {
            "structured": struct.astype(np.float32),
            "cnn_frame": cnn
        }
        
        return obs, {}

    def step(self, action):
        """
        Execute one environment step with SAC continuous action.
        
        Args:
            action: np.ndarray of shape (7,) from SAC policy
        """
        action_dict = self._decode_sac_action(action)
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
            
            # ── Pain Penalty (Health Drop) [DISABLED FOR NOW] ───────────────────
            if player_health < self.prev_health and not self.is_dead:
                pass 
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
            
            # ── Bank UI & Follow UI Penalty & Auto-Close ─────────────────────────────
            current_time = time.time()
            if not self.is_dead and (current_time - getattr(self, 'last_bank_check_time', 0.0) >= 1.0):
                self.last_bank_check_time = current_time
                
                # Check for Bank UI
                ui_coords = self.game_detector.detect_bank_ui(frame)
                is_bank = True
                
                # Check for Follow UI if Bank UI isn't open
                if ui_coords is None:
                    ui_coords = self.game_detector.detect_follow_ui(frame)
                    is_bank = False
                
                if ui_coords is not None:
                    step_reward -= 5.0
                    ui_name = "BANK" if is_bank else "FOLLOW"
                    print(f"  ❌ {ui_name} UI OPENED! -5.0 penalty applied. Auto-closing...")
                    
                    # Convert internal 800x600 coordinates to actual screen coordinates
                    # The capture region starts at (192, 156) for the 1536x888 window
                    # This math maps the template center to the actual screen pixel
                    capture_cfg = self.config.get('capture', {})
                    monitor = capture_cfg.get('monitor', 1)
                    if monitor == 1: # Assuming primary monitor fullscreen/borderless
                        # Wait, input controller already has move_mouse_absolute which expects 0-1
                        # Force click uses absolute raw pixels for PyDirectInput
                        # The bounding box of Roblox is captured by mss
                        from feature_extraction.screen_processor import ScreenCapture
                        # Calculate precise absolute screen coordinates by adding the capture region offset
                        # Since the frame is exactly the capture resolution, no scaling is needed.
                        screen_x = self.screen_capture.monitor['left'] + int(ui_coords[0])
                        screen_y = self.screen_capture.monitor['top'] + int(ui_coords[1])
                        # To click UI in Roblox, we must temporarily break out of Shift Lock
                        self.input_controller.press_key('SHIFT', duration=0.1)
                        time.sleep(0.1) # Wait for Roblox to unlock cursor
                        self.input_controller.force_click(screen_x, screen_y)
                        time.sleep(0.1)
                        self.input_controller.press_key('SHIFT', duration=0.1) # Re-engage Shift Lock
            
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
                    else:
                        # Leaving again after a re-entry → offset the re-entry penalty to prevent infinite penalty loop
                        step_reward += abs(self.reward_safe_zone_reenter)
                        print(f"  🏃 LEFT SAFE ZONE AGAIN! +{abs(self.reward_safe_zone_reenter)} reward")
                
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
            if self.step_count % 15 == 0:
                action_keys = action_dict.get('keys', [])
                clicking = action_dict.get('click_left', False)
                mouse_dx = action_dict.get('mouse_dx', 0)
                print(f"  [Step {self.step_count}] Enemies: {len(enemies)} | "
                      f"Safe: {in_safe_zone} | HP: {player_health:.0%} | "
                      f"Keys: {action_keys} | Click: {clicking} | "
                      f"Mouse: {mouse_dx:.2f} | "
                      f"Reward: {self.episode_reward:.2f}")
                      
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
                
                try:
                    bgr_frame = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Roblox Bot Vision", bgr_frame)
                    # Force window to stay on top so it floats above Roblox
                    cv2.setWindowProperty("Roblox Bot Vision", cv2.WND_PROP_TOPMOST, 1)
                    cv2.waitKey(1)
                except cv2.error:
                    # OpenCV built without GUI support — disable vision window
                    self.show_vision = False
                    print("⚠️  cv2.imshow not available. Disabling debug vision window.")

            if terminated:
                break
                
            if self.step_count >= self.config.get('steps_per_episode', 500):
                truncated = True
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
