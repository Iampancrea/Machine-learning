import cv2
import time
from feature_extraction.color_detector import GameDetector
from feature_extraction.screen_processor import ScreenCapture

def test_enemy_vision():
    print("📸 Initializing real-time vision test...")
    # Initialize to capture the FULL Roblox window (1536x888)
    screen_cap = ScreenCapture(resolution=(1536, 888), region=None)
    detector = GameDetector(config={})
    
    print("⏳ Starting loop... Bring Roblox into focus!")
    print("👉 Press 'q' inside the vision window to quit.")
    time.sleep(2)
    
    cv2.namedWindow("Roblox Bot Full Vision Test", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Roblox Bot Full Vision Test", cv2.WND_PROP_TOPMOST, 1)
    
    while True:
        frame = screen_cap.capture()
        enemies = detector.detect_enemies(frame)
        
        # Draw annotations on a copy of the frame to keep it clean
        annotated_frame = frame.copy()
        
        for i, enemy in enumerate(enemies):
            # Draw green circle at the estimated player center
            px, py = enemy['player_center']
            cv2.circle(annotated_frame, (px, py), 6, (0, 255, 0), -1)
            
            # Draw blue circle at the tag center (HP text)
            if 'tag_center' in enemy:
                tx, ty = enemy['tag_center']
                cv2.circle(annotated_frame, (tx, ty), 4, (255, 0, 0), -1)
            
            # Draw red bounding box around the HP text blob
            if 'hp_bar' in enemy:
                box = enemy['hp_bar']
                if len(box) == 4:
                    x, y, w, h = box
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Label it
            cv2.putText(annotated_frame, f"Enemy {i+1}", (px + 10, py), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        # cv2 expects BGR for rendering
        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Roblox Bot Full Vision Test", bgr_frame)
        
        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    print("👋 Vision test closed.")

if __name__ == "__main__":
    test_enemy_vision()
