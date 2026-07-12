"""
Data collection module - records gameplay and saves state-action pairs.

UPDATED: Now saves raw 80x60 grayscale CNN frames alongside structured
features in the .npz files. This is essential for training the CNN branch
of the hybrid model on cloud GPUs (Kaggle, SageMaker, Lightning AI).

Uses GameDetector for geometric-anchored enemy detection and safe zone ROI.
"""
import numpy as np
import cv2
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle

from feature_extraction.screen_processor import ScreenCapture
from feature_extraction.color_detector import GameDetector
from feature_extraction.feature_engineer import FeatureEngineer


class DataRecorder:
    """Record gameplay data for behavior cloning training"""
    
    def __init__(self, save_dir: str = "./data/recordings",
                 config: dict = None):
        """
        Initialize data recorder
        
        Args:
            save_dir: Directory to save recordings
            config: Configuration dictionary
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        
        # Initialize screen capture
        self.screen_capture = ScreenCapture(
            resolution=tuple(self.config.get('capture', {}).get('resolution', [800, 600])),
            fps=self.config.get('capture', {}).get('fps', 30)
        )
        
        # Initialize game-specific detector (geometric anchoring)
        self.game_detector = GameDetector(config=self.config)
        
        # Initialize feature engineer (structured features + CNN frame prep)
        self.feature_engineer = FeatureEngineer(
            history_length=self.config.get('features', {}).get('feature_history_length', 10),
            cnn_resolution=tuple(self.config.get('features', {}).get('cnn_resolution', [80, 60]))
        )
        
        # Recording state
        self.is_recording = False
        self.session_data = []
        self.current_session_id = None
        
    def start_session(self, session_name: str = None) -> str:
        """
        Start a new recording session
        
        Args:
            session_name: Optional custom name for the session
            
        Returns:
            Session ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = session_name or f"session_{timestamp}"
        self.session_data = []
        self.is_recording = True
        self.feature_engineer.reset()
        
        print(f"Started recording session: {self.current_session_id}")
        return self.current_session_id
    
    def record_frame(self, action: Dict) -> Dict:
        """
        Record a single frame with associated action.
        
        Captures the screen, runs GameDetector for enemy detection + safe zone,
        extracts structured features AND the 80x60 grayscale CNN frame.
        
        Args:
            action: Dictionary containing keyboard/mouse actions
                   e.g., {'keys': ['W', 'A'], 'mouse_dx': 0.5, 'mouse_dy': -0.2, 'click': False}
                   
        Returns:
            Recorded data dictionary
        """
        if not self.is_recording:
            raise RuntimeError("No active recording session. Call start_session() first.")
        
        # Capture frame
        frame = self.screen_capture.capture()
        
        # Run game-specific detection (geometric anchoring)
        enemies = self.game_detector.detect_enemies(frame)
        in_safe_zone = self.game_detector.detect_safe_zone(frame)
        
        # Extract structured features + CNN frame
        structured_features, cnn_frame = self.feature_engineer.extract_features(
            frame=frame,
            enemies=enemies,
            in_safe_zone=in_safe_zone,
            game_state=None
        )
        
        # Create recording entry
        timestamp = time.time()
        record = {
            'timestamp': timestamp,
            'features': structured_features,     # 1D float32 array
            'cnn_frame': cnn_frame,              # 2D float32 (60, 80) — for CNN training
            'action': action,
            'enemy_detected': len(enemies) > 0,
            'enemy_count': len(enemies),
            'in_safe_zone': in_safe_zone,
        }
        
        self.session_data.append(record)
        return record
    
    def stop_session(self, save: bool = True) -> str:
        """
        Stop recording and optionally save data
        
        Args:
            save: Whether to save the recorded data
            
        Returns:
            Path to saved file or None
        """
        self.is_recording = False
        
        if not save or len(self.session_data) == 0:
            print(f"Session ended. {len(self.session_data)} frames recorded (not saved).")
            return None
        
        # Save data
        save_path = self.save_dir / f"{self.current_session_id}.npz"
        
        # Convert to numpy arrays
        features_list = [r['features'] for r in self.session_data]
        cnn_frames_list = [r['cnn_frame'] for r in self.session_data]
        timestamps = np.array([r['timestamp'] for r in self.session_data])
        enemy_detected = np.array([r['enemy_detected'] for r in self.session_data])
        enemy_counts = np.array([r['enemy_count'] for r in self.session_data])
        in_safe_zone = np.array([r['in_safe_zone'] for r in self.session_data])
        
        # Save actions as JSON strings
        actions_json = [json.dumps(r['action']) for r in self.session_data]
        
        np.savez_compressed(
            save_path,
            features=np.array(features_list),           # (N, structured_dim)
            cnn_frames=np.array(cnn_frames_list),        # (N, 60, 80) — CNN training data
            timestamps=timestamps,                       # (N,)
            enemy_detected=enemy_detected,               # (N,) bool
            enemy_counts=enemy_counts,                   # (N,) int
            in_safe_zone=in_safe_zone,                   # (N,) bool
            actions=np.array(actions_json, dtype=object)  # (N,) JSON strings
        )
        
        print(f"\nSession saved to: {save_path}")
        print(f"Total frames: {len(self.session_data)}")
        print(f"Duration: {timestamps[-1] - timestamps[0]:.2f}s")
        
        # Size report
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"  ├─ features: {np.array(features_list).nbytes / 1024:.1f} KB")
        print(f"  └─ cnn_frames: {np.array(cnn_frames_list).nbytes / (1024*1024):.1f} MB")
        
        return str(save_path)
    
    def get_statistics(self) -> Dict:
        """Get statistics about current session"""
        if len(self.session_data) == 0:
            return {}
        
        enemy_count = sum(1 for r in self.session_data if r['enemy_detected'])
        safe_count = sum(1 for r in self.session_data if r['in_safe_zone'])
        duration = self.session_data[-1]['timestamp'] - self.session_data[0]['timestamp']
        
        return {
            'total_frames': len(self.session_data),
            'frames_with_enemy': enemy_count,
            'enemy_detection_rate': enemy_count / len(self.session_data),
            'frames_in_safe_zone': safe_count,
            'safe_zone_rate': safe_count / len(self.session_data),
            'duration_seconds': duration,
            'avg_fps': len(self.session_data) / max(duration, 0.001)
        }


class ActionLogger:
    """Log keyboard and mouse inputs during gameplay"""
    
    def __init__(self):
        """Initialize action logger"""
        self.current_keys = set()
        self.mouse_delta = (0.0, 0.0)
        self.mouse_click = False
        
    def press_key(self, key: str):
        """Record key press"""
        self.current_keys.add(key.upper())
        
    def release_key(self, key: str):
        """Record key release"""
        self.current_keys.discard(key.upper())
        
    def move_mouse(self, dx: float, dy: float):
        """Record mouse movement"""
        self.mouse_delta = (dx, dy)
        
    def click(self, button: str = 'left'):
        """Record mouse click"""
        self.mouse_click = True
        
    def get_action(self) -> Dict:
        """Get current action state"""
        action = {
            'keys': list(self.current_keys),
            'mouse_dx': self.mouse_delta[0],
            'mouse_dy': self.mouse_delta[1],
            'click': self.mouse_click
        }
        
        # Reset transient states
        self.mouse_delta = (0.0, 0.0)
        self.mouse_click = False
        
        return action
    
    def reset(self):
        """Reset all action states"""
        self.current_keys.clear()
        self.mouse_delta = (0.0, 0.0)
        self.mouse_click = False


if __name__ == "__main__":
    print("Data Recorder Test")
    print("=" * 50)
    print("This module records gameplay for training data.")
    print("\nNow saves BOTH structured features AND raw 80x60")
    print("grayscale CNN frames in .npz files for cloud training.")
    print("\nTo use:")
    print("1. Configure your game colors in the constructor")
    print("2. Call start_session()")
    print("3. Call record_frame(action) for each frame")
    print("4. Call stop_session() to save")
    print("\nNote: This requires running alongside Roblox with pynput for input capture")
