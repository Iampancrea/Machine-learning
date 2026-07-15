"""
Screen capture module using DXCam for massive FPS gains
Optimized for low-latency capture on Windows machines

FIXED: Switched from MSS to DXCam as requested in the audit.
"""
import numpy as np
import cv2
import dxcam
import pydirectinput as pdi
from typing import Tuple, Optional
import time


class ScreenCapture:
    """High-performance screen capture for Roblox gameplay using DXCam"""
    
    def __init__(self, resolution: Tuple[int, int] = (800, 600), 
                 region: Optional[Tuple[int, int, int, int]] = None,
                 fps: int = 30):
        """
        Initialize screen capture
        
        Args:
            resolution: Capture bounding box size (width, height)
            region: Explicit screen region (left, top, width, height).
                    If None, an 800x600 box is centered on the primary monitor.
            fps: Target frames per second
        """
        self.target_resolution = resolution
        self.region = region
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        # Initialize DXCam
        self.camera = dxcam.create(output_color="RGB")
        
        # Build the capture bounding box
        if region is not None:
            self.monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3],
            }
        else:
            # Center an (width × height) box on the primary monitor
            screen_w, screen_h = pdi.size()
            cap_w, cap_h = self.target_resolution
            center_x = screen_w // 2
            center_y = screen_h // 2
            self.monitor = {
                "left": center_x - cap_w // 2,
                "top": center_y - cap_h // 2,
                "width": cap_w,
                "height": cap_h,
            }
            
        self.dxcam_region = (
            self.monitor["left"],
            self.monitor["top"],
            self.monitor["left"] + self.monitor["width"],
            self.monitor["top"] + self.monitor["height"]
        )
        
        print(f"Capturing region (DXCam): {self.dxcam_region}")
        
        self.last_capture_time = 0
        self.frame_count = 0
        self.last_frame = None
        
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
        
        # Grab frame with DXCam
        frame = self.camera.grab(region=self.dxcam_region)
        
        # DXCam returns None if the screen hasn't updated.
        if frame is None:
            if self.last_frame is not None:
                return self.last_frame
            else:
                # Force wait for the first frame
                while frame is None:
                    time.sleep(0.005)
                    frame = self.camera.grab(region=self.dxcam_region)
                    
        self.last_frame = frame
        return frame
    
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
        print(f"Frame shape: {frame.shape if frame is not None else 'None'}")
        print(f"Bounding box: {self.dxcam_region}")


if __name__ == "__main__":
    # Test the screen capture with strict 800x600 bounding box
    capture = ScreenCapture(resolution=(800, 600))
    capture.test_capture(duration=5.0)
