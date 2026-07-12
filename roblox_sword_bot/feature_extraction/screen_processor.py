"""
Screen capture module using MSS (fast screenshot library)
Optimized for low-latency capture on integrated graphics

FIXED: Uses mss.MSS() (non-deprecated) and captures a strict bounding box
at the OS level instead of grabbing full-screen and resizing after.
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
    
    def __init__(self, resolution: Tuple[int, int] = (800, 600), 
                 region: Optional[Tuple[int, int, int, int]] = None,
                 fps: int = 30):
        """
        Initialize screen capture
        
        Args:
            resolution: Capture bounding box size (width, height) — grabbed
                        directly at the OS level. No post-capture resize.
            region: Explicit screen region (left, top, width, height).
                    If None, an 800x600 box is centered on the primary monitor.
            fps: Target frames per second
        """
        self.target_resolution = resolution
        self.region = region
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        # Initialize MSS — use the non-deprecated class constructor
        self.sct = mss.MSS()
        
        # Build the capture bounding box
        if region is not None:
            # Explicit region supplied by caller
            self.monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3],
            }
        else:
            # Center an (width × height) box on the primary monitor
            primary = self.sct.monitors[1]  # monitors[0] is "all", [1] is primary
            cap_w, cap_h = self.target_resolution
            center_x = primary["left"] + primary["width"] // 2
            center_y = primary["top"] + primary["height"] // 2
            self.monitor = {
                "left": center_x - cap_w // 2,
                "top": center_y - cap_h // 2,
                "width": cap_w,
                "height": cap_h,
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
        
        # Capture screenshot — already the exact bounding box, no resize needed
        screenshot = self.sct.grab(self.monitor)
        
        # Convert to numpy array (BGRA format)
        img = np.array(screenshot)
        
        # Remove alpha channel (convert BGRA to BGR)
        img_bgr = img[:, :, :3]
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
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
        print(f"Bounding box: {self.monitor}")


if __name__ == "__main__":
    # Test the screen capture with strict 800x600 bounding box
    capture = ScreenCapture(resolution=(800, 600))
    capture.test_capture(duration=5.0)
