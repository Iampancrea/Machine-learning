"""
Color-based detection for finding players and enemies in Roblox
Optimized for CPU execution on integrated graphics
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional


class ColorDetector:
    """Detect players/enemies based on color signatures"""
    
    def __init__(self, target_colors: List[Tuple[int, int, int]], 
                 tolerance: int = 30):
        """
        Initialize color detector
        
        Args:
            target_colors: List of RGB colors to detect [(R,G,B), ...]
            tolerance: Color matching tolerance (0-255)
        """
        self.target_colors = [np.array(color) for color in target_colors]
        self.tolerance = tolerance
        
    def create_mask(self, frame: np.ndarray, color_idx: int = None) -> np.ndarray:
        """
        Create binary mask for target colors
        
        Args:
            frame: Input RGB frame
            color_idx: Specific color index or None for all colors
            
        Returns:
            Binary mask where detected colors are white
        """
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        masks = []
        
        colors_to_check = [color_idx] if color_idx is not None else range(len(self.target_colors))
        
        for idx in colors_to_check:
            color_bgr = cv2.cvtColor(
                np.uint8([[self.target_colors[idx]]]), 
                cv2.COLOR_RGB2BGR
            )[0][0]
            
            # Convert target color to HSV
            color_hsv = cv2.cvtColor(
                np.uint8([[color_bgr]]), 
                cv2.COLOR_BGR2HSV
            )[0][0]
            
            h, s, v = color_hsv
            
            # Define color range
            lower_color = np.array([
                max(0, h - self.tolerance // 4),
                max(0, s - self.tolerance),
                max(0, v - self.tolerance)
            ])
            
            upper_color = np.array([
                min(180, h + self.tolerance // 4),
                min(255, s + self.tolerance),
                min(255, v + self.tolerance)
            ])
            
            # Create mask
            mask = cv2.inRange(hsv, lower_color, upper_color)
            masks.append(mask)
        
        # Combine all masks
        if len(masks) > 1:
            combined_mask = cv2.bitwise_or(masks[0], masks[1])
            for mask in masks[2:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        else:
            combined_mask = masks[0]
        
        # Morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        return combined_mask
    
    def find_contours(self, mask: np.ndarray, 
                     min_area: int = 50) -> List[Tuple[int, int, int, int]]:
        """
        Find contours in mask and return bounding boxes
        
        Args:
            mask: Binary mask from create_mask()
            min_area: Minimum contour area to consider
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))
        
        return boxes
    
    def detect_players(self, frame: np.ndarray) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Detect all players in frame
        
        Args:
            frame: Input RGB frame
            
        Returns:
            Dictionary with color indices as keys and list of bounding boxes as values
        """
        results = {}
        
        for idx, color in enumerate(self.target_colors):
            mask = self.create_mask(frame, idx)
            boxes = self.find_contours(mask)
            results[f"color_{idx}"] = boxes
        
        return results
    
    def get_nearest_enemy(self, frame: np.ndarray, 
                         screen_center: Tuple[int, int] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the nearest enemy to screen center
        
        Args:
            frame: Input RGB frame
            screen_center: Center point reference (defaults to image center)
            
        Returns:
            Nearest bounding box (x, y, w, h) or None
        """
        if screen_center is None:
            screen_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        
        all_boxes = []
        
        # Get all detected boxes
        for idx in range(len(self.target_colors)):
            mask = self.create_mask(frame, idx)
            boxes = self.find_contours(mask)
            all_boxes.extend(boxes)
        
        if not all_boxes:
            return None
        
        # Find nearest to center
        min_distance = float('inf')
        nearest_box = None
        
        for box in all_boxes:
            x, y, w, h = box
            box_center = (x + w // 2, y + h // 2)
            distance = np.sqrt(
                (box_center[0] - screen_center[0])**2 + 
                (box_center[1] - screen_center[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_box = box
        
        return nearest_box


if __name__ == "__main__":
    # Test color detection
    print("Color detector initialized")
    print("Target colors should be configured based on your Roblox game")
    print("Common Roblox character colors:")
    print("  - Bright blue: (0, 107, 167)")
    print("  - Bright red: (205, 0, 0)")
    print("  - Yellow: (255, 255, 0)")
    print("  - Green: (0, 255, 0)")
