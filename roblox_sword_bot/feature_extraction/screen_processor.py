"""
Screen capture module using MSS (fast screenshot library)
Optimized for low-latency capture on integrated graphics
"""
import numpy as np
import cv2
from PIL import Image
import mss
import mss.tools
from typing import Tuple, Optional
import time


class ScreenCapture:
    """High-performance screen capture for Roblox gameplay"""
    
    def __init__(self, resolution: Tuple[int, int] = (320, 180), 
                 region: Optional[Tuple[int, int, int, int]] = None,
                 fps: int = 30):
        """
        Initialize screen capture
        
        Args:
            resolution: Target resolution (width, height) - downscaled for speed
            region: Screen region to capture (x, y, width, height) or None for full screen
            fps: Target frames per second
        """
        self.target_resolution = resolution
        self.region = region
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        # Initialize MSS
        self.sct = mss.mss()
        
        # Get monitor info
        if region is None:
            # Full screen - use primary monitor
            self.monitor = self.sct.monitors[0]  # Full screen
            print(f"Capturing full screen: {self.monitor}")
        else:
            # Custom region
            self.monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3]
            }
            print(f"Capturing region: {self.monitor}")
        
        self.last_capture_time = 0
        self.frame_count = 0
        
    def capture(self) -> np.ndarray:
        """
        Capture and process a single frame
        
        Returns:
            numpy array of shape (height, width, 3) in RGB format
        """
        # Frame rate limiting
        current_time = time.time()
        elapsed = current_time - self.last_capture_time
        
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        
        self.last_capture_time = time.time()
        self.frame_count += 1
        
        # Capture screenshot
        screenshot = self.sct.grab(self.monitor)
        
        # Convert to numpy array (BGRA format)
        img = np.array(screenshot)
        
        # Remove alpha channel (convert BGRA to BGR)
        img_bgr = img[:, :, :3]
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Downscale to target resolution
        if self.target_resolution:
            img_rgb = cv2.resize(
                img_rgb, 
                self.target_resolution, 
                interpolation=cv2.INTER_AREA  # Best quality for downscaling
            )
        
        return img_rgb
    
    def capture_batch(self, num_frames: int) -> np.ndarray:
        """
        Capture multiple frames
        
        Args:
            num_frames: Number of frames to capture
            
        Returns:
            numpy array of shape (num_frames, height, width, 3)
        """
        frames = []
        for _ in range(num_frames):
            frame = self.capture()
            frames.append(frame)
        
        return np.stack(frames, axis=0)
    
    def get_fps(self) -> float:
        """Get current capture FPS"""
        return self.frame_count / (time.time() - self.last_capture_time + 1e-6)
    
    def test_capture(self, duration: float = 5.0):
        """Test capture performance"""
        print(f"Testing capture for {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            frame = self.capture()
            
        elapsed = time.time() - start_time
        avg_fps = self.frame_count / elapsed
        
        print(f"Captured {self.frame_count} frames in {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Frame shape: {frame.shape}")
        print(f"Target resolution: {self.target_resolution}")


if __name__ == "__main__":
    # Test the screen capture
    capture = ScreenCapture(resolution=(320, 180))
    capture.test_capture(duration=5.0)
