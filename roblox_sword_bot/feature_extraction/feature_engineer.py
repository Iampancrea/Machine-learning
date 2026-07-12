"""
Feature engineering module - combines all extracted features into a single vector
Reduces 1920x1080 pixels (~2M values) to ~20 meaningful features
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class FeatureEngineer:
    """Extract and combine game state features for ML model input"""
    
    def __init__(self, history_length: int = 10):
        """
        Initialize feature engineer
        
        Args:
            history_length: Number of past frames to include in features
        """
        self.history_length = history_length
        self.feature_history = deque(maxlen=history_length)
        
        # Feature dimensions
        self.num_features = self._calculate_feature_dim()
        
    def _calculate_feature_dim(self) -> int:
        """Calculate total feature vector size"""
        # Per-frame features + historical features
        base_features = 15  # enemy position, distance, health, etc.
        return base_features + (self.history_length - 1) * 5  # Only track key features historically
    
    def extract_features(self, frame: np.ndarray, 
                        enemy_box: Optional[Tuple[int, int, int, int]],
                        game_state: Optional[Dict] = None) -> np.ndarray:
        """
        Extract features from a single frame
        
        Args:
            frame: Current game frame (RGB)
            enemy_box: Enemy bounding box (x, y, w, h) or None
            game_state: Additional game state info (health, cooldown, etc.)
            
        Returns:
            Feature vector (1D numpy array)
        """
        features = []
        
        # Screen dimensions
        height, width = frame.shape[:2]
        screen_center = (width // 2, height // 2)
        
        # === Enemy Features ===
        if enemy_box is not None:
            x, y, w, h = enemy_box
            
            # Enemy position relative to screen center
            enemy_center_x = x + w // 2
            enemy_center_y = y + h // 2
            
            # Normalized position (-1 to 1)
            rel_x = (enemy_center_x - screen_center[0]) / (width // 2)
            rel_y = (enemy_center_y - screen_center[1]) / (height // 2)
            
            features.extend([
                rel_x,  # Enemy X offset
                rel_y,  # Enemy Y offset
                w / width,  # Enemy width (indicates distance)
                h / height,  # Enemy height
                w * h / (width * height),  # Enemy area ratio
                np.sqrt(rel_x**2 + rel_y**2),  # Distance to enemy
                np.arctan2(rel_y, rel_x),  # Angle to enemy
            ])
            
            # Enemy velocity (if we have history)
            if len(self.feature_history) > 0:
                prev_features = self.feature_history[-1]
                vel_x = rel_x - prev_features[0]
                vel_y = rel_y - prev_features[1]
                features.extend([vel_x, vel_y])  # Enemy velocity
            else:
                features.extend([0.0, 0.0])
                
        else:
            # No enemy detected - use default values
            features.extend([0.0] * 9)
        
        # === Player Features (from game state or estimation) ===
        if game_state is not None:
            features.extend([
                game_state.get('health', 1.0),  # Normalized health
                game_state.get('cooldown', 0.0),  # Attack cooldown (0-1)
                game_state.get('stamina', 1.0),  # Normalized stamina
            ])
        else:
            features.extend([1.0, 0.0, 1.0])  # Defaults
        
        # === Frame-level Features ===
        # Average brightness (can indicate lighting conditions)
        avg_brightness = np.mean(frame) / 255.0
        features.append(avg_brightness)
        
        # Color distribution (simplified)
        color_std = np.std(frame, axis=(0, 1)).mean() / 255.0
        features.append(color_std)
        
        # Store features in history
        self.feature_history.append(np.array(features))
        
        # Add historical features
        if len(self.feature_history) > 1:
            # Add simplified history (only key features)
            for i in range(1, min(len(self.feature_history), self.history_length)):
                hist_feat = self.feature_history[-(i+1)]
                # Only add important historical features
                features.extend([
                    hist_feat[0],  # Historical enemy X
                    hist_feat[1],  # Historical enemy Y
                    hist_feat[5],  # Historical distance
                ])
        
        # Pad if history is short
        while len(features) < self.num_features:
            features.append(0.0)
        
        return np.array(features[:self.num_features], dtype=np.float32)
    
    def reset(self):
        """Clear feature history"""
        self.feature_history.clear()
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features for debugging"""
        names = [
            'enemy_rel_x', 'enemy_rel_y', 'enemy_width', 'enemy_height',
            'enemy_area', 'enemy_distance', 'enemy_angle',
            'enemy_vel_x', 'enemy_vel_y',
            'player_health', 'attack_cooldown', 'player_stamina',
            'frame_brightness', 'color_variance'
        ]
        
        # Add historical feature names
        for i in range(1, self.history_length):
            names.extend([
                f'hist_{i}_enemy_x',
                f'hist_{i}_enemy_y',
                f'hist_{i}_distance'
            ])
        
        return names
    
    def get_observation_space_shape(self) -> Tuple[int]:
        """Return shape of observation space for RL"""
        return (self.num_features,)


if __name__ == "__main__":
    # Test feature extraction
    engineer = FeatureEngineer(history_length=5)
    
    # Simulate a frame
    frame = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
    enemy_box = (100, 50, 40, 80)  # Example enemy box
    game_state = {'health': 0.8, 'cooldown': 0.2, 'stamina': 0.9}
    
    features = engineer.extract_features(frame, enemy_box, game_state)
    
    print(f"Feature vector shape: {features.shape}")
    print(f"Number of features: {engineer.num_features}")
    print(f"Feature names: {engineer.get_feature_names()}")
    print(f"Sample features: {features[:10]}")
