"""
Data collection module - records gameplay and saves state-action pairs
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
from feature_extraction.color_detector import ColorDetector
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
        
        # Initialize components
        self.screen_capture = ScreenCapture(
            resolution=tuple(self.config.get('capture', {}).get('resolution', [800, 600])),
            fps=self.config.get('capture', {}).get('fps', 30)
        )
        
        # Initialize color detector with default Roblox colors
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
        Record a single frame with associated action
        
        Args:
            action: Dictionary containing keyboard/mouse actions
                   e.g., {'keys': ['W', 'A'], 'mouse_x': 0.5, 'mouse_y': -0.2, 'click': False}
                   
        Returns:
            Recorded data dictionary
        """
        if not self.is_recording:
            raise RuntimeError("No active recording session. Call start_session() first.")
        
        # Capture frame
        frame = self.screen_capture.capture()
        
        # Detect enemies
        enemy_box = self.color_detector.get_nearest_enemy(frame)
        
        # Extract features
        features = self.feature_engineer.extract_features(
            frame=frame,
            enemy_box=enemy_box,
            game_state=None  # Can be extended to read from game
        )
        
        # Create recording entry
        timestamp = time.time()
        record = {
            'timestamp': timestamp,
            'features': features,
            'action': action,
            'enemy_detected': enemy_box is not None,
            'enemy_box': enemy_box,
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
        timestamps = np.array([r['timestamp'] for r in self.session_data])
        enemy_detected = np.array([r['enemy_detected'] for r in self.session_data])
        
        # Save actions as JSON-serializable format
        actions_json = [json.dumps(r['action']) for r in self.session_data]
        
        np.savez_compressed(
            save_path,
            features=np.array(features_list),
            timestamps=timestamps,
            enemy_detected=enemy_detected,
            actions=np.array(actions_json, dtype=object)
        )
        
        print(f"Session saved to: {save_path}")
        print(f"Total frames: {len(self.session_data)}")
        print(f"Duration: {timestamps[-1] - timestamps[0]:.2f}s")
        
        return str(save_path)
    
    def get_statistics(self) -> Dict:
        """Get statistics about current session"""
        if len(self.session_data) == 0:
            return {}
        
        enemy_count = sum(1 for r in self.session_data if r['enemy_detected'])
        duration = self.session_data[-1]['timestamp'] - self.session_data[0]['timestamp']
        
        return {
            'total_frames': len(self.session_data),
            'frames_with_enemy': enemy_count,
            'enemy_detection_rate': enemy_count / len(self.session_data),
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
    print("\nTo use:")
    print("1. Configure your game colors in the constructor")
    print("2. Call start_session()")
    print("3. Call record_frame(action) for each frame")
    print("4. Call stop_session() to save")
    print("\nNote: This requires running alongside Roblox with pynput for input capture")
