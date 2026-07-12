"""
Feature engineering module - combines detector outputs + CNN frame preparation
into a unified feature vector for the hybrid model.

NO OCR. The 80x60 grayscale CNN branch handles spatial/UI awareness.
Structured features come from GameDetector's geometric anchoring.
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import deque


class FeatureEngineer:
    """Extract and combine game state features for the hybrid ML model"""
    
    def __init__(self, history_length: int = 10,
                 cnn_resolution: Tuple[int, int] = (80, 60)):
        """
        Initialize feature engineer
        
        Args:
            history_length: Number of past frames to include in features
            cnn_resolution: (width, height) for CNN input frame
        """
        self.history_length = history_length
        self.cnn_resolution = cnn_resolution  # (width, height)
        self.feature_history = deque(maxlen=history_length)
        
        # Per-frame structured feature count (BEFORE history)
        self._base_feature_count = 11
        # Historical features tracked per past frame
        self._hist_features_per_frame = 3
        
    def get_structured_dim(self) -> int:
        """
        Return total structured feature vector dimension.
        This must match what the HybridNetwork expects as structured_dim.
        """
        return (self._base_feature_count + 
                (self.history_length - 1) * self._hist_features_per_frame)
    
    def prepare_cnn_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Downscale RGB frame to 80x60 grayscale for CNN branch.
        
        Args:
            frame: Full-resolution RGB frame (e.g. 800x600x3)
            
        Returns:
            Grayscale frame normalized to 0-1, shape (60, 80), dtype float32
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, self.cnn_resolution, 
                          interpolation=cv2.INTER_AREA)
        return small  # Return raw uint8 array
    
    def extract_features(self, frame: np.ndarray,
                        enemies: List[dict],
                        in_safe_zone: bool,
                        game_state: Optional[Dict] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract structured features + CNN frame from a single game tick.
        
        Args:
            frame: Current game frame (RGB, e.g. 800x600x3)
            enemies: List of confirmed enemy dicts from GameDetector.detect_enemies()
                     Each dict has: hp_bar, hp_pct, player_center, tag_center, text_confidence
            in_safe_zone: Whether player is currently in the safe zone
            game_state: Optional additional state (reserved for future use)
            
        Returns:
            Tuple of (structured_features, cnn_frame)
            - structured_features: 1D float32 numpy array
            - cnn_frame: 2D float32 numpy array of shape (60, 80)
        """
        features = []
        
        height, width = frame.shape[:2]
        screen_center = (width // 2, height // 2)
        
        # ─── Enemy Features (nearest enemy) ───
        if enemies:
            # Find nearest enemy to screen center
            nearest = min(enemies, key=lambda e:
                np.sqrt((e['player_center'][0] - screen_center[0]) ** 2 +
                        (e['player_center'][1] - screen_center[1]) ** 2))
            
            px, py = nearest['player_center']
            
            # Relative position normalized to (-1, 1)
            rel_x = (px - screen_center[0]) / (width // 2)
            rel_y = (py - screen_center[1]) / (height // 2)
            distance = np.sqrt(rel_x ** 2 + rel_y ** 2)
            angle = np.arctan2(rel_y, rel_x)
            
            features.extend([
                rel_x,                      # 0: nearest enemy X offset
                rel_y,                      # 1: nearest enemy Y offset
                distance,                   # 2: distance to nearest enemy
                angle,                      # 3: angle to nearest enemy
                nearest['hp_pct'],          # 4: nearest enemy HP (0-1)
                float(len(enemies)),        # 5: total visible enemy count
            ])
            
            # Enemy velocity (frame-to-frame delta)
            if len(self.feature_history) > 0:
                prev = self.feature_history[-1]
                vel_x = rel_x - prev[0]
                vel_y = rel_y - prev[1]
                features.extend([vel_x, vel_y])  # 6-7: velocity
            else:
                features.extend([0.0, 0.0])
        else:
            # No enemies visible — zero out all 8 enemy features
            features.extend([0.0] * 8)
        
        # ─── Zone Features ───
        features.append(1.0 if in_safe_zone else 0.0)  # 8: safe zone flag
        
        # ─── Frame-level Features ───
        avg_brightness = np.mean(frame) / 255.0
        
        # Build outputs first so we can use cnn_frame for fast variance calc
        cnn_frame = self.prepare_cnn_frame(frame)
        color_variance = np.std(cnn_frame) / 255.0
        
        features.extend([avg_brightness, color_variance])  # 9-10
        
        # ─── Store base features in history ───
        base_features = np.array(features[:self._base_feature_count],
                                 dtype=np.float32)
        self.feature_history.append(base_features)
        
        # ─── Historical Features (past N-1 frames) ───
        for i in range(1, self.history_length):
            if i < len(self.feature_history):
                hist = self.feature_history[-(i + 1)]
                features.extend([
                    hist[0],   # historical enemy X
                    hist[1],   # historical enemy Y
                    hist[2],   # historical distance
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # ─── Build outputs ───
        total_dim = self.get_structured_dim()
        structured = np.array(features[:total_dim], dtype=np.float32)
        
        # Pad if somehow short (shouldn't happen, but safety)
        if len(structured) < total_dim:
            structured = np.pad(structured, (0, total_dim - len(structured)))
        
        return structured, cnn_frame
    
    def reset(self):
        """Clear feature history"""
        self.feature_history.clear()
    
    def get_feature_names(self) -> List[str]:
        """Get names of all structured features for debugging"""
        names = [
            'enemy_rel_x', 'enemy_rel_y', 'enemy_distance', 'enemy_angle',
            'enemy_hp_pct', 'enemy_count',
            'enemy_vel_x', 'enemy_vel_y',
            'in_safe_zone',
            'frame_brightness', 'color_variance',
        ]
        
        for i in range(1, self.history_length):
            names.extend([
                f'hist_{i}_enemy_x',
                f'hist_{i}_enemy_y',
                f'hist_{i}_distance',
            ])
        
        return names
    
    def get_observation_space_shape(self) -> Tuple[int]:
        """Return shape of structured observation space for RL"""
        return (self.get_structured_dim(),)


if __name__ == "__main__":
    # Test feature extraction
    engineer = FeatureEngineer(history_length=5)
    
    print(f"Structured feature dim: {engineer.get_structured_dim()}")
    print(f"CNN resolution: {engineer.cnn_resolution}")
    print(f"Feature names: {engineer.get_feature_names()}")
    
    # Simulate a frame with one enemy
    frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    enemies = [{
        'hp_bar': (350, 200, 60, 8),
        'hp_pct': 0.75,
        'player_center': (380, 258),
        'tag_center': (380, 204),
        'text_confidence': 0.12,
    }]
    
    structured, cnn_frame = engineer.extract_features(
        frame, enemies, in_safe_zone=False
    )
    
    print(f"\nStructured features shape: {structured.shape}")
    print(f"CNN frame shape: {cnn_frame.shape}")
    print(f"CNN frame dtype: {cnn_frame.dtype}")
    print(f"CNN frame range: [{cnn_frame.min():.3f}, {cnn_frame.max():.3f}]")
    print(f"Sample structured: {structured[:6]}")
