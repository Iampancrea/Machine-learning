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
                    'mouse_dy': mouse_dy * 0.5,
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
        Predict action from features
        
        Args:
            structured_features: 1D feature vector from FeatureEngineer
            cnn_frame: 2D grayscale frame (60, 80) from FeatureEngineer
            
        Returns:
            Tuple of (action_dict, confidence)
        """
        with torch.no_grad():
            struct_tensor = torch.FloatTensor(structured_features).unsqueeze(0).to(self.device)
            
            if self.is_hybrid:
                # CNN frame: add batch and channel dims → (1, 1, 60, 80)
                cnn_tensor = torch.FloatTensor(cnn_frame).unsqueeze(0).unsqueeze(0).to(self.device)
                outputs = self.model(struct_tensor, cnn_tensor)
            else:
                outputs = self.model(struct_tensor)
            
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            action_idx = predicted.item()
            conf = confidence.item()
            
            if conf < self.confidence_threshold:
                return {'keys': [], 'mouse_dx': 0, 'mouse_dy': 0, 'click_left': False, 'click_right': False, 'click': False}, conf
            
            action = self._decode_action(action_idx)
            return action, conf
    
    def run(self):
        """Main bot loop"""
        print("\n🤖 Bot starting...")
        print("Press ESC to hard-kill the process at any time\n")
        
        _start_esc_kill_switch()
        
        self.running = True
        self.feature_engineer.reset()
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                # Capture frame
                frame = self.screen_capture.capture()
                
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
                
                # Execute action (skip if in safe zone for safety)
                if confidence >= self.confidence_threshold and not in_safe_zone:
                    self.input_controller.execute_action(action)
                
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
