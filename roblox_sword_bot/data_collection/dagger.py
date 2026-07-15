"""
Dataset Aggregation (DAgger) loop
Runs the bot but records your expert actions whenever you take control.
"""
import time
import numpy as np
import torch
import pynput.keyboard
import pynput.mouse
from pathlib import Path
from typing import Dict, Any

from inference.bot_controller import BotController
from data_collection.recorder import DataRecorder, ActionLogger

class DAggerLoop:
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.bot = BotController(model_path, config)
        
        # Override bot's input execution so it only moves if we aren't overriding
        self.user_is_overriding = False
        
        # Recorder logic
        self.recorder = DataRecorder(config)
        self.action_logger = ActionLogger()
        self.session_id = None
        
    def _setup_listeners(self):
        def on_key_press(key):
            self.user_is_overriding = True
            try:
                self.action_logger.press_key(key.char if hasattr(key, 'char') and key.char else str(key))
            except: pass
            
        def on_key_release(key):
            try:
                self.action_logger.release_key(key.char if hasattr(key, 'char') and key.char else str(key))
            except: pass
            
        def on_click(x, y, button, pressed):
            self.user_is_overriding = True
            if pressed:
                if button == pynput.mouse.Button.left:
                    self.action_logger.click('left')
                elif button == pynput.mouse.Button.right:
                    self.action_logger.click('right')
                    
        def on_move(x, y):
            if hasattr(on_move, 'last_x'):
                import pydirectinput as pdi
                sw, sh = pdi.size()
                dx = (x - on_move.last_x) / float(sw)
                dy = (y - on_move.last_y) / float(sh)
                
                # If mouse moved significantly, user is overriding
                if abs(dx) > 0.001 or abs(dy) > 0.001:
                    self.user_is_overriding = True
                    self.action_logger.move_mouse(dx, dy)
            on_move.last_x = x
            on_move.last_y = y
            
        self.kb_listener = pynput.keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        self.mouse_listener = pynput.mouse.Listener(on_click=on_click, on_move=on_move)
        
    def run(self, duration: int = 1800):
        print("\n🗡️ Starting DAgger Loop...")
        print("The bot will play normally. If you touch the keyboard or mouse, it will yield control to you.")
        print("Your expert actions will be recorded to correct its mistakes.")
        
        self.bot.screen_capture = self.recorder.screen_capture
        self._setup_listeners()
        self.kb_listener.start()
        self.mouse_listener.start()
        
        self.session_id = self.recorder.start_session()
        
        start_time = time.time()
        last_frame_time = start_time
        frame_interval = 1.0 / self.config.get('inference', {}).get('fps_limit', 30)
        
        try:
            while time.time() - start_time < duration:
                current_time = time.time()
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                
                last_frame_time = current_time
                self.user_is_overriding = False # reset every frame
                
                # Let bot think
                frame = self.bot.screen_capture.capture()
                
                # Update detectors
                if self.bot.config.get('features', {}).get('use_color_detection', True):
                    enemies, safe_zone = self.bot.game_detector.detect_enemies(frame)
                else:
                    enemies, safe_zone = [], False
                
                struct_feat, cnn_frame = self.bot.feature_engineer.extract_features(
                    frame, enemies, safe_zone, 1.0, None, None
                )
                
                bot_action, confidence = self.bot.predict_action(struct_feat, cnn_frame)
                
                # Check user action
                user_action = self.action_logger.get_action()
                
                # If user is overriding, record USER action, execute USER action (which happens physically anyway)
                if self.user_is_overriding:
                    self.recorder.record_frame(user_action)
                else:
                    # Execute bot action if confident
                    if confidence >= self.bot.confidence_threshold and not safe_zone:
                        self.bot.input_controller.execute_action(bot_action)
                    else:
                        self.bot.input_controller.execute_action({'keys': [], 'mouse_dx': 0, 'mouse_dy': 0, 'click_left': False, 'click_right': False})
                        
        except KeyboardInterrupt:
            print("\n⏹️ DAgger stopped.")
            
        finally:
            self.kb_listener.stop()
            self.mouse_listener.stop()
            self.recorder.stop_session(save=True)
            print("DAgger dataset saved.")
