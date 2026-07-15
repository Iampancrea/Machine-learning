"""
Bot Controller - Runs the trained hybrid model to play Roblox automatically.

Uses GameDetector (geometric anchoring) for enemy detection, FeatureEngineer
for structured features + CNN frame prep, and HybridNetwork for action prediction.

ESC kill-switch armed via pynput + os._exit(0).
"""
import torch
import numpy as np
import time
import os
import threading
from pathlib import Path
from typing import Dict, Optional

import pynput.keyboard

from utils.config import load_config
from feature_extraction.screen_processor import ScreenCapture
from feature_extraction.color_detector import GameDetector
from feature_extraction.feature_engineer import FeatureEngineer
from models.network import MLPNetwork, HybridNetwork, create_model
from utils.input_control import InputController, _start_esc_kill_switch


class BotController:
    """Main controller for running the trained bot"""
    
    def __init__(self, model_path: str, config: dict = None):
        """
        Initialize bot controller
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration dictionary
        """
        self.config = config or {}
        self.device = self.config.get('hardware', {}).get('device', 'cpu')
        
        print(f"Loading model from: {model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Check if this is a Decision Transformer (no dict wrapper, or specific key)
        self.is_dt = 'transformer.layers.0.self_attn.in_proj_weight' in checkpoint or 'pos_emb.weight' in checkpoint
        
        if self.is_dt:
            from models.decision_transformer import DecisionTransformer
            self.model = DecisionTransformer(
                struct_dim=15, 
                cnn_channels=self.config.get('features', {}).get('cnn_frame_stack', 4) + 1,
                action_dim=9,
                max_length=self.config.get('dt', {}).get('context_len', 32)
            ).to(self.device)
            self.model.load_state_dict(checkpoint)
            
            # Context buffers for autoregressive inference
            self.dt_context_len = self.config.get('dt', {}).get('context_len', 32)
            self.target_rtg = self.config.get('dt', {}).get('target_rtg', 1.0)
            
            self.struct_buf = []
            self.cnn_buf = []
            self.act_buf = []
            self.rtg_buf = []
            self.is_hybrid = False
            
        else:
            self.num_actions = checkpoint['num_actions']
            self.action_mapping = checkpoint.get('action_mapping', None)
            model_type = checkpoint.get('model_type', 'mlp')
            
            # Initialize model architecture based on checkpoint type
            if model_type == 'hybrid':
                structured_dim = checkpoint.get('structured_dim', 38)
                self.model = HybridNetwork(
                    structured_dim=structured_dim,
                    cnn_output_dim=self.config.get('model', {}).get('cnn_output_dim', 32),
                    hidden_layers=self.config.get('model', {}).get('hidden_layers', [128, 64]),
                    output_dim=self.num_actions,
                    dropout=0.0  # No dropout at inference
                ).to(self.device)
                self.is_hybrid = True
            else:
                feature_history = self.config.get('features', {}).get('feature_history_length', 10)
                input_dim = checkpoint.get('structured_dim', feature_history * 15)
                self.model = MLPNetwork(
                    input_dim=input_dim,
                    hidden_layers=self.config.get('model', {}).get('hidden_layers', [256, 128, 64]),
                    output_dim=self.num_actions
                ).to(self.device)
                self.is_hybrid = False
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        print(f"Model loaded successfully ({sum(p.numel() for p in self.model.parameters()):,} parameters)")
        if self.is_dt:
            print("Model type: Decision Transformer (Autoregressive Sequence)")
        else:
            print(f"Model type: {'hybrid (structured + CNN)' if self.is_hybrid else 'MLP only'}")
        
        # Initialize components
        self.screen_capture = ScreenCapture(
            resolution=tuple(self.config.get('capture', {}).get('resolution', [800, 600])),
            fps=self.config.get('capture', {}).get('fps', 30)
        )
        
        self.game_detector = GameDetector(config=self.config)
        
        self.feature_engineer = FeatureEngineer(
            history_length=self.config.get('features', {}).get('feature_history_length', 10),
            cnn_resolution=tuple(self.config.get('features', {}).get('cnn_resolution', [80, 60]))
        )
        
        self.input_controller = InputController(config=self.config)
        
        # Action decoding
        self.reverse_action_mapping = {}
        if self.action_mapping:
            self.reverse_action_mapping = {v: k for k, v in self.action_mapping.items()}
        
        # Inference settings
        self.fps_limit = self.config.get('inference', {}).get('fps_limit', 60)
        self.confidence_threshold = self.config.get('inference', {}).get('confidence_threshold', 0.7)
        self.safety_mode = self.config.get('inference', {}).get('safety_mode', True)
        
        self.running = False
    
    def _decode_action(self, action_idx: int) -> Dict:
        """Convert action index to keyboard/mouse commands"""
        
        if self.reverse_action_mapping and action_idx in self.reverse_action_mapping:
            action_str = self.reverse_action_mapping[action_idx]
            parts = action_str.split('_')
            
            if len(parts) >= 5:
                keys = parts[0].split(',') if parts[0] else []
                click_left = parts[1] == '1'
                click_right = parts[2] == '1'
                mouse_dx = int(parts[3])
                mouse_dy = int(parts[4])
                
                return {
                    'keys': keys,
                    'mouse_dx': mouse_dx * 0.5,
                    'mouse_dy': mouse_dy * 0.05,  # Restrict vertical tilt to keep camera level
                    'click_left': click_left,
                    'click_right': click_right
                }
        
        # Default fallback
        action_map = {
            0: {'keys': [], 'mouse_dx': 0, 'mouse_dy': 0, 'click_left': False, 'click_right': False},
            1: {'keys': ['W'], 'mouse_dx': 0, 'mouse_dy': 0, 'click_left': False, 'click_right': False},
            2: {'keys': ['A'], 'mouse_dx': 0, 'mouse_dy': 0, 'click_left': False, 'click_right': False},
            3: {'keys': ['S'], 'mouse_dx': 0, 'mouse_dy': 0, 'click_left': False, 'click_right': False},
            4: {'keys': ['D'], 'mouse_dx': 0, 'mouse_dy': 0, 'click_left': False, 'click_right': False},
            5: {'keys': ['W', 'A'], 'mouse_dx': 0, 'mouse_dy': 0, 'click_left': False, 'click_right': False},
            6: {'keys': ['W', 'D'], 'mouse_dx': 0, 'mouse_dy': 0, 'click_left': False, 'click_right': False},
            7: {'keys': [], 'mouse_dx': 0, 'mouse_dy': 0, 'click_left': True, 'click_right': False},
        }
        
        return action_map.get(action_idx % 8, action_map[0])
    
    def predict_action(self, structured_features: np.ndarray,
                      cnn_frame: np.ndarray) -> tuple:
        """
        Predict action from features (with MC-Dropout Uncertainty Estimation)
        
        Args:
            structured_features: 1D feature vector from FeatureEngineer
            cnn_frame: 2D grayscale frame (5, 60, 80) from FeatureEngineer
            
        Returns:
            Tuple of (action_dict, confidence)
        """
        with torch.no_grad():
            if self.is_dt:
                # Append current state to buffer
                struct_tensor = torch.FloatTensor(structured_features).to(self.device)
                cnn_tensor = torch.FloatTensor(cnn_frame).to(self.device)
                
                # Append current RTG target
                rtg_tensor = torch.FloatTensor([self.target_rtg]).to(self.device)
                
                # If first step, use zeros for previous action
                if not self.act_buf:
                    act_tensor = torch.zeros(9).to(self.device)
                else:
                    act_tensor = self.act_buf[-1] # previous action
                    
                self.struct_buf.append(struct_tensor)
                self.cnn_buf.append(cnn_tensor)
                self.act_buf.append(act_tensor)
                self.rtg_buf.append(rtg_tensor)
                
                # Truncate to context length
                if len(self.struct_buf) > self.dt_context_len:
                    self.struct_buf.pop(0)
                    self.cnn_buf.pop(0)
                    self.act_buf.pop(0)
                    self.rtg_buf.pop(0)
                    
                # Stack sequences
                s_seq = torch.stack(self.struct_buf)
                c_seq = torch.stack(self.cnn_buf)
                a_seq = torch.stack(self.act_buf)
                r_seq = torch.stack(self.rtg_buf)
                
                action = self.model.get_action(s_seq, c_seq, a_seq, r_seq, deterministic=True)
                
                # Apply base scaling for gameplay
                action['mouse_dx'] *= 0.5
                action['mouse_dy'] *= 0.05
                
                # For DT, we overwrite the last recorded action with the actual predicted action 
                # for the next step's input to be accurate
                
                new_act = torch.zeros(9).to(self.device)
                key_map = {'W': 0, 'A': 1, 'S': 2, 'D': 3, 'Space': 4}
                for k in action['keys']:
                    if k in key_map:
                        new_act[key_map[k]] = 1.0
                new_act[5] = 1.0 if action['click_left'] else 0.0
                new_act[6] = 1.0 if action['click_right'] else 0.0
                new_act[7] = float(action['mouse_dx'] / 0.5)
                new_act[8] = float(action['mouse_dy'] / 0.05)
                self.act_buf[-1] = new_act
                
                # DT confidence is assumed high, though we could measure entropy
                confidence = 1.0
                return action, confidence

            # --- Legacy BC Inference (Hybrid/MLP) ---
            struct_tensor = torch.FloatTensor(structured_features).unsqueeze(0).to(self.device)
            
            if self.is_hybrid:
                cnn_tensor = torch.FloatTensor(cnn_frame).unsqueeze(0).to(self.device)
                
                # --- MC-Dropout for Uncertainty Estimation ---
                self.model.train()  # Enable dropout
                mc_runs = 3
                mouse_preds = []
                for _ in range(mc_runs):
                    _, _, m_out = self.model(struct_tensor, cnn_tensor)
                    mouse_preds.append(m_out)
                
                self.model.eval()  # Back to inference mode
                
                stacked = torch.stack(mouse_preds)
                variance = torch.var(stacked, dim=0).mean().item()
                
                # High variance = low confidence (OOD). Max variance for Tanh is ~1.0
                confidence = max(0.0, 1.0 - (variance * 2.0))
                
                # Get actual action
                action = self.model.get_action(struct_tensor, cnn_tensor, deterministic=True)
                
                # Apply base scaling for gameplay (as requested in Quick Wins)
                action['mouse_dx'] *= 0.5
                action['mouse_dy'] *= 0.05
                
                return action, confidence
            else:
                # Legacy MLP Support
                outputs = self.model(struct_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
                action_idx = predicted.item()
                conf = confidence.item()
                return self._decode_action(action_idx), conf
    
    def run(self):
        """Main bot loop"""
        print("\n🤖 Bot starting...")
        print("Press ESC to hard-kill the process at any time\n")
        
        _start_esc_kill_switch()
        
        self.running = True
        self.feature_engineer.reset()
        
        frame_count = 0
        start_time = time.time()
        last_bank_check_time = 0.0
        
        try:
            while self.running:
                # Capture frame
                frame = self.screen_capture.capture()
                current_time = time.time()
                
                # Detect and handle UI popups (auto-close) - throttled to save CPU
                if current_time - last_bank_check_time >= 1.0:
                    last_bank_check_time = current_time
                    
                    # Check for Bank UI
                    ui_coords = self.game_detector.detect_bank_ui(frame)
                    is_bank = True
                    
                    # Check for Follow UI if Bank UI isn't open
                    if ui_coords is None:
                        ui_coords = self.game_detector.detect_follow_ui(frame)
                        is_bank = False
                    
                    if ui_coords is not None:
                        ui_name = "BANK" if is_bank else "FOLLOW"
                        print(f"\n  ❌ {ui_name} UI OPENED! Auto-closing...", flush=True)
                        screen_x = self.screen_capture.monitor['left'] + int(ui_coords[0])
                        screen_y = self.screen_capture.monitor['top'] + int(ui_coords[1])
                        self.input_controller.press_key('SHIFT', duration=0.1)
                        time.sleep(0.1)
                        self.input_controller.force_click(screen_x, screen_y)
                        time.sleep(0.1)
                        self.input_controller.press_key('SHIFT', duration=0.1)
                        # Don't execute normal actions this frame
                        continue
                
                # Detect enemies (geometric anchoring)
                enemies = self.game_detector.detect_enemies(frame)
                in_safe_zone = self.game_detector.detect_safe_zone(frame)
                
                # Extract features (structured + CNN frame)
                structured_features, cnn_frame = self.feature_engineer.extract_features(
                    frame=frame,
                    enemies=enemies,
                    in_safe_zone=in_safe_zone,
                    game_state=None
                )
                
                # Predict action
                action, confidence = self.predict_action(structured_features, cnn_frame)
                
                # Execute action (release keys if in safe zone for safety)
                if confidence >= self.confidence_threshold:
                    if not in_safe_zone:
                        self.input_controller.execute_action(action)
                    else:
                        self.input_controller.execute_action({'keys': [], 'mouse_dx': 0, 'mouse_dy': 0, 'click_left': False, 'click_right': False})
                
                # Update statistics
                frame_count += 1
                elapsed = time.time() - start_time
                
                if frame_count % 30 == 0:
                    fps = frame_count / max(elapsed, 0.001)
                    status = "SAFE" if in_safe_zone else ("AIMING" if enemies else "SEARCHING")
                    enemy_str = f"👥{len(enemies)}" if enemies else "👥0"
                    action_str = f"Keys:{action['keys']} L:{action['click_left']}"
                    print(f"\r[{status}] {enemy_str} | FPS: {fps:.1f} | Conf: {confidence:.3f} | Action: {action_str}",
                          end='', flush=True)
                
                # Frame rate limiting
                frame_time = 1.0 / self.fps_limit
                current_frame_time = time.time() - start_time - (frame_count - 1) * frame_time
                if current_frame_time < frame_time:
                    time.sleep(frame_time - current_frame_time)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping bot...")
            self.running = False
        finally:
            self.input_controller.reset()
            total_time = time.time() - start_time
            avg_fps = frame_count / max(total_time, 0.001)
            print(f"\nSession ended. Total frames: {frame_count}, Avg FPS: {avg_fps:.1f}")
    
    def stop(self):
        """Stop the bot"""
        self.running = False


if __name__ == "__main__":
    print("Bot Controller Test")
    print("=" * 50)
    print("This module runs the trained hybrid bot.")
    print("\nTo use:")
    print("1. Train a model first: python main.py train_bc")
    print("2. Run the bot: python main.py run --model checkpoints/best_model.pth")
    print("\n⚠️  WARNING: This will control your keyboard and mouse!")
    print("🔑  Press ESC at any time to hard-kill the process.")
