import cv2
import time
from feature_extraction.color_detector import GameDetector
from feature_extraction.screen_processor import ScreenCapture

def test_enemy_vision():
    print("📸 Initializing vision test...")
    # Initialize to capture the FULL Roblox window (1536x888)
    screen_cap = ScreenCapture(resolution=(1536, 888), region=None)
    detector = GameDetector(config={})
    
    print("⏳ Waiting 3 seconds... Bring Roblox into focus!")
    time.sleep(3)
    
    print("📸 Capturing frame...")
    frame = screen_cap.capture()
    
    print("🔍 Detecting enemies...")
    enemies = detector.detect_enemies(frame)
    
    print(f"✅ Found {len(enemies)} enemies!")
    
    # Draw annotations
    for i, enemy in enumerate(enemies):
        # Draw green circle at the estimated player center
        px, py = enemy['player_center']
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
        
        # Draw blue circle at the tag center (HP text)
        tx, ty = enemy['tag_center']
        cv2.circle(frame, (tx, ty), 3, (255, 0, 0), -1)
        
        # Draw red bounding box around the HP text blob
        box = enemy['hp_bar']
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Label it
        cv2.putText(frame, f"Enemy {i+1}", (px + 10, py), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
    # Save the result
    output_path = "vision_test_result.jpg"
    # frame is RGB from ScreenCapture, cv2 expects BGR for saving
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_frame)
    print(f"💾 Saved annotated image to: {output_path}")
    print("👉 Please open this file to verify if the bot is seeing the enemies correctly!")

if __name__ == "__main__":
    test_enemy_vision()
