import time
import sys
from feature_extraction.screen_processor import ScreenCapture
from feature_extraction.color_detector import GameDetector
from utils.input_control import InputController
from utils.config import load_config

def main():
    print("🏦 Bank UI Auto-Close Tester")
    print("=" * 30)
    print("Loading config...")
    cfg = load_config()
    
    print("Initializing components...")
    screen_capture = ScreenCapture(
        resolution=tuple(cfg.config.get('capture', {}).get('resolution', [1536, 888])),
        fps=10 # Lower FPS for testing UI is fine
    )
    game_detector = GameDetector(config=cfg.config)
    input_controller = InputController(config=cfg.config)
    
    print("\n✅ Ready! Switch to Roblox.")
    print("Press Ctrl+C in this terminal to stop.")
    print("Waiting for Bank UI to appear on screen...\n")
    
    try:
        while True:
            # Capture full screen based on config
            frame = screen_capture.capture()
            
            # Detect Bank UI
            bank_coords = game_detector.detect_bank_ui(frame)
            
            if bank_coords is not None:
                screen_x = screen_capture.monitor['left'] + int(bank_coords[0])
                screen_y = screen_capture.monitor['top'] + int(bank_coords[1])
                
                print(f"🎯 BANK UI DETECTED at ({bank_coords[0]}, {bank_coords[1]}) in frame!")
                print(f"   Absolute Screen Coords: ({screen_x}, {screen_y})")
                print("   Auto-closing in 0.5s (giving you a moment to see it)...")
                time.sleep(0.5)
                
                # Auto-close logic
                print("   Executing click...")
                input_controller.press_key('SHIFT', duration=0.1)
                time.sleep(0.1)
                input_controller.force_click(screen_x, screen_y)
                time.sleep(0.1)
                input_controller.press_key('SHIFT', duration=0.1)
                print("   Done! Waiting 2 seconds before resuming scanning...\n")
                
                time.sleep(2.0)
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user. Good testing, handsome!")
        sys.exit(0)

if __name__ == "__main__":
    main()
