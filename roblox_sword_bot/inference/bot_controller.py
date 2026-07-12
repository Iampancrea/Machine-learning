"""
Bot Controller - Runs the trained model to play Roblox automatically
"""
import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional

from utils.config import load_config
from feature_extraction.screen_processor import ScreenCapture
from feature_extraction.color_detector import ColorDetector
from feature_extraction.feature_engineer import FeatureEngineer
from models.network import MLPNetwork
from utils.input_control import InputController


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
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_state = checkpoint['model_state_dict']
        self.num_actions = checkpoint['num_actions']
        self.action_mapping = checkpoint.get('action_mapping', None)
        
        # Initialize model architecture
        # Note: We need to know input_dim from somewhere - using config estimate
        feature_history = self.config.get('features', {}).get('feature_history_length', 10)
        input_dim = feature_history * 15  # Approximate
        
        self.model = MLPNetwork(
            input_dim=input_dim,
            hidden_layers=self.config.get('model', {}).get('hidden_layers', [256, 128, 64]),
            output_dim=self.num_actions
        ).to(self.device)
        
        self.model.load_state_dict(self.model_state)
        self.model.eval()
        
        print(f"Model loaded successfully ({self.model.num_params:,} parameters)")
        
        # Initialize components
        self.screen_capture = ScreenCapture(
            resolution=tuple(self.config.get('capture', {}).get('resolution', [320, 180])),
            fps=self.config.get('capture', {}).get('fps', 30)
        )
        
        default_colors = [
            (0, 107, 167),    # Bright blue
            (205, 0, 0),      # Bright red
            (255, 255, 0),    # Yellow
            (0, 255, 0),      # Green
        ]
        self.color_detector = ColorDetector(
            target_colors=default_colors,
            tolerance=self.config.get('features', {}).get('color_tolerance', 30)
        )
        
        self.feature_engineer = FeatureEngineer(
            history_length=self.config.get('features', {}).get('feature_history_length', 10)
        )
        
        self.input_controller = InputController(config=self.config)
        
        # Action decoding (will be populated if available in checkpoint)
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
        
        # If we have reverse mapping, use it
        if self.reverse_action_mapping and action_idx in self.reverse_action_mapping:
            action_str = self.reverse_action_mapping[action_idx]
            parts = action_str.split('_')
            
            if len(parts) >= 4:
                keys = parts[0].split(',') if parts[0] else []
                click = parts[1] == '1'
                mouse_dx = int(parts[2])
                mouse_dy = int(parts[3])
                
                return {
                    'keys': keys,
                    'mouse_dx': mouse_dx * 0.5,  # Scale down
                    'mouse_dy': mouse_dy * 0.5,
                    'click': click
                }
        
        # Default fallback - map action indices to simple actions
        action_map = {
            0: {'keys': [], 'mouse_dx': 0, 'mouse_dy': 0, 'click': False},
            1: {'keys': ['W'], 'mouse_dx': 0, 'mouse_dy': 0, 'click': False},
            2: {'keys': ['A'], 'mouse_dx': 0, 'mouse_dy': 0, 'click': False},
            3: {'keys': ['S'], 'mouse_dx': 0, 'mouse_dy': 0, 'click': False},
            4: {'keys': ['D'], 'mouse_dx': 0, 'mouse_dy': 0, 'click': False},
            5: {'keys': ['W', 'A'], 'mouse_dx': 0, 'mouse_dy': 0, 'click': False},
            6: {'keys': ['W', 'D'], 'mouse_dx': 0, 'mouse_dy': 0, 'click': False},
            7: {'keys': [], 'mouse_dx': 0, 'mouse_dy': 0, 'click': True},
        }
        
        return action_map.get(action_idx % 8, action_map[0])
    
    def predict_action(self, features: np.ndarray) -> tuple:
        """
        Predict action from features
        
        Args:
            features: Feature vector
            
        Returns:
            Tuple of (action_dict, confidence)
        """
        with torch.no_grad():
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get model output
            outputs = self.model(features_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Get best action
            confidence, predicted = torch.max(probs, 1)
            action_idx = predicted.item()
            conf = confidence.item()
            
            # Apply confidence threshold
            if conf < self.confidence_threshold:
                # Return no-op action if confidence too low
                return {'keys': [], 'mouse_dx': 0, 'mouse_dy': 0, 'click': False}, conf
            
            # Decode action
            action = self._decode_action(action_idx)
            
            return action, conf
    
    def run(self):
        """Main bot loop"""
        print("\n🤖 Bot starting...")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        self.feature_engineer.reset()
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                # Capture frame
                frame = self.screen_capture.capture()
                
                # Detect enemies
                enemy_box = self.color_detector.get_nearest_enemy(frame)
                
                # Extract features
                features = self.feature_engineer.extract_features(
                    frame=frame,
                    enemy_box=enemy_box,
                    game_state=None
                )
                
                # Predict action
                action, confidence = self.predict_action(features)
                
                # Execute action
                if confidence >= self.confidence_threshold:
                    self.input_controller.execute_action(action)
                
                # Update statistics
                frame_count += 1
                elapsed = time.time() - start_time
                
                if frame_count % 30 == 0:
                    fps = frame_count / max(elapsed, 0.001)
                    status = "AIMING" if enemy_box else "SEARCHING"
                    print(f"\r[{status}] FPS: {fps:.1f} | Conf: {confidence:.3f}", end='', flush=True)
                
                # Frame rate limiting
                frame_time = 1.0 / self.fps_limit
                current_frame_time = time.time() - start_time - (frame_count - 1) * frame_time
                if current_frame_time < frame_time:
                    time.sleep(frame_time - current_frame_time)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping bot...")
            self.running = False
        finally:
            # Cleanup
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
    print("This module runs the trained bot.")
    print("\nTo use:")
    print("1. Train a model first: python main.py train_bc")
    print("2. Run the bot: python main.py run --model checkpoints/best_model.pth")
    print("\n⚠️  WARNING: This will control your keyboard and mouse!")
