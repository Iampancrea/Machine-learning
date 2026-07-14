"""
Keyboard and mouse control module using pydirectinput-rgx and Pynput
Includes human-like delays and anti-detection features

FIXED: pyautogui replaced with pydirectinput_rgx for DirectInput
       compatibility in 3D games (Roblox, etc.)
FIXED: Async pynput ESC listener → os._exit(0) hard kill-switch.
"""
import pydirectinput as pdi
pdi.FAILSAFE = False  # Disable corner-of-screen mouse crash
import pynput.keyboard
import pynput.mouse
import numpy as np
import time
import random
import os
import threading
from typing import List, Tuple, Optional, Dict


# ─────────────────────────────────────────────────────────────────────
#  GLOBAL ESC KILL-SWITCH
#  Spawns a daemon thread that listens for the Escape key at all times.
#  When pressed, os._exit(0) bypasses all cleanup and nukes the process.
# ─────────────────────────────────────────────────────────────────────
import _thread

_kill_switch_active = False
_kill_switch_lock = threading.Lock()
_last_esc_time = 0.0


def _start_esc_kill_switch():
    """Start the global ESC kill-switch listener (idempotent)."""
    global _kill_switch_active
    with _kill_switch_lock:
        if _kill_switch_active:
            return
        _kill_switch_active = True

    def _on_press(key):
        global _last_esc_time
        if key == pynput.keyboard.Key.esc:
            now = time.time()
            if now - _last_esc_time < 2.0:
                print("\n\n🛑  ESC DOUBLE-PRESSED — HARD KILLING PROCESS IMMEDIATELY!")
                os._exit(0)
            else:
                _last_esc_time = now
                print("\n\n🛑  ESC PRESSED — Requesting graceful save and exit...")
                print("💡 (Press ESC again within 2 seconds to force close without saving)")
                _thread.interrupt_main()

    listener = pynput.keyboard.Listener(on_press=_on_press)
    listener.daemon = True
    listener.start()
    print("🔑  ESC kill-switch armed (press ESC to terminate; double-press to hard exit)")


class InputController:
    """Control keyboard and mouse with human-like behavior"""
    
    def __init__(self, config: dict = None):
        """
        Initialize input controller
        
        Args:
            config: Configuration dictionary with humanization settings
        """
        self.config = config or {}
        
        # Human-like behavior settings
        self.enable_delay = self.config.get('humanization', {}).get('enable_reaction_delay', False)
        self.min_reaction_time = self.config.get('humanization', {}).get('min_reaction_time_ms', 150) / 1000.0
        self.max_reaction_time = self.config.get('humanization', {}).get('max_reaction_time_ms', 350) / 1000.0
        self.jitter_std = self.config.get('humanization', {}).get('jitter_std', 0.05)
        
        self.left_held = False
        self.right_held = False
        self.action_randomness = self.config.get('humanization', {}).get('action_randomness', 0.02)
        
        # Mouse settings
        self.sensitivity = self.config.get('actions', {}).get('mouse_sensitivity', 0.5)
        self.screen_width, self.screen_height = pdi.size()
        
        # Keyboard state
        self.pressed_keys = set()
        
        # Setup keyboard listener (optional, for manual override)
        self.keyboard_listener = None
        self.mouse_listener = None
        
        # pydirectinput-rgx has no global failsafe toggle — that's fine,
        # our ESC kill-switch replaces that safety net entirely.
        
        # Arm the ESC kill-switch on first InputController creation
        _start_esc_kill_switch()
        
    def _apply_delay(self):
        """Apply human-like reaction delay"""
        if self.enable_delay:
            delay = random.uniform(self.min_reaction_time, self.max_reaction_time)
            # Add small jitter
            delay += np.random.normal(0, self.jitter_std * 0.1)
            delay = max(0.01, delay)  # Minimum 10ms
            time.sleep(delay)
    
    def _add_jitter(self, value: float) -> float:
        """Add small random variation to make movements less robotic"""
        jitter = np.random.normal(0, self.jitter_std)
        return value + jitter
    
    def press_key(self, key: str, duration: float = None):
        """
        Press a keyboard key
        
        Args:
            key: Key to press (e.g., 'w', 'a', 'space')
            duration: How long to hold the key (None = until release)
        """
        self._apply_delay()
        
        # Normalize key name for pydirectinput-rgx
        key = key.upper()
        if key.startswith('KEY.'):
            key = key[4:]
            
        key_map = {
            'SPACE': 'space',
            'SHIFT': 'shift',
            'SHIFT_L': 'shift',
            'SHIFT_R': 'shift',
            'CTRL': 'ctrl',
            'CTRL_L': 'ctrl',
            'CTRL_R': 'ctrl',
            'ALT': 'alt',
            'ALT_L': 'alt',
            'ALT_R': 'alt',
        }
        pdi_key = key_map.get(key, key.lower())
        
        pdi.keyDown(pdi_key)
        self.pressed_keys.add(key)
        
        if duration:
            time.sleep(duration)
            self.release_key(key)
    
    def release_key(self, key: str):
        """
        Release a keyboard key
        
        Args:
            key: Key to release
        """
        key = key.upper()
        if key.startswith('KEY.'):
            key = key[4:]
            
        key_map = {
            'SPACE': 'space',
            'SHIFT': 'shift',
            'SHIFT_L': 'shift',
            'SHIFT_R': 'shift',
            'CTRL': 'ctrl',
            'CTRL_L': 'ctrl',
            'CTRL_R': 'ctrl',
            'ALT': 'alt',
            'ALT_L': 'alt',
            'ALT_R': 'alt',
        }
        pdi_key = key_map.get(key, key.lower())
        
        try:
            pdi.keyUp(pdi_key)
            self.pressed_keys.discard(key)
        except:
            pass  # Key might not be pressed
    
    def execute_keys(self, keys: List[str]):
        """
        Execute a set of keys (press missing ones, release extra ones)
        
        Args:
            keys: List of keys that should be pressed
        """
        keys_upper = [k.upper() for k in keys]
        
        # Release keys that shouldn't be pressed
        for key in list(self.pressed_keys):
            if key not in keys_upper:
                self.release_key(key)
        
        # Press keys that should be pressed
        for key in keys_upper:
            if key not in self.pressed_keys:
                self.press_key(key)
    
    def move_mouse_relative(self, dx: float, dy: float, 
                           duration: float = 0.05):
        """
        Move mouse relative to current position
        
        Args:
            dx: Horizontal movement (-1 to 1 normalized)
            dy: Vertical movement (-1 to 1 normalized)
            duration: Movement duration in seconds (ignored by pdi, kept for API compat)
        """
        self._apply_delay()
        
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return
            
        # Denormalize (recorder divides by 800 and 600)
        pixel_dx = int(dx * self.sensitivity * 800)
        pixel_dy = int(dy * self.sensitivity * 600)
        
        # pydirectinput-rgx moveRel — no tween support, raw DirectInput
        pdi.moveRel(
            pixel_dx, 
            pixel_dy, 
            relative=True,
        )
    
    def move_mouse_absolute(self, x: float, y: float, 
                           duration: float = 0.1):
        """
        Move mouse to absolute position
        
        Args:
            x: X position (0-1 normalized)
            y: Y position (0-1 normalized)
            duration: Movement duration in seconds (ignored by pdi, kept for API compat)
        """
        self._apply_delay()
        
        # Add jitter
        x = self._add_jitter(x)
        y = self._add_jitter(y)
        
        # Clamp to screen bounds
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        
        # Convert to pixels
        pixel_x = int(x * self.screen_width)
        pixel_y = int(y * self.screen_height)
        
        pdi.moveTo(pixel_x, pixel_y)
    
    def click(self, button: str = 'left', clicks: int = 1):
        """
        Click mouse button
        
        Args:
            button: Button to click ('left', 'right', 'middle')
            clicks: Number of clicks
        """
        self._apply_delay()
        
        for _ in range(clicks):
            pdi.click(button=button)
            if clicks > 1:
                time.sleep(0.1)  # Delay between multiple clicks
    
    def force_click(self, x: int, y: int):
        """
        Instantly move the mouse to absolute screen coordinates and click.
        Used to forcefully close UI popups (like the Bank Menu).
        """
        pdi.moveTo(x, y)
        time.sleep(0.05)
        pdi.click()
        print(f"🖱️ Force-clicked at ({x}, {y})")

    def execute_action(self, action: Dict):
        """
        Execute a complete action from model output
        
        Args:
            action: Dictionary with keys:
                   - 'keys': List of keys to press
                   - 'mouse_dx': Mouse horizontal movement
                   - 'mouse_dy': Mouse vertical movement
                   - 'click_left': Whether to click/hold left click
                   - 'click_right': Whether to click/hold right click
        """
        # Execute keyboard actions
        if 'keys' in action:
            self.execute_keys(action['keys'])
        
        # Execute right click (RMB) hold/release for camera rotation when not in Shift Lock
        click_right = action.get('click_right', False)
        if click_right and not self.right_held:
            pdi.mouseDown(button='right')
            self.right_held = True
        elif not click_right and self.right_held:
            pdi.mouseUp(button='right')
            self.right_held = False
            
        # Execute mouse movement
        if 'mouse_dx' in action and 'mouse_dy' in action:
            self.move_mouse_relative(
                action['mouse_dx'],
                action['mouse_dy']
            )
        
        # Execute left click (M1)
        click_left = action.get('click_left', action.get('click', False))
        if click_left and not self.left_held:
            pdi.mouseDown(button='left')
            self.left_held = True
        elif not click_left and self.left_held:
            pdi.mouseUp(button='left')
            self.left_held = False
    
    def reset(self):
        """Release all keys and reset state"""
        for key in list(self.pressed_keys):
            self.release_key(key)
        self.pressed_keys.clear()
        if self.left_held:
            pdi.mouseUp(button='left')
            self.left_held = False
        if self.right_held:
            pdi.mouseUp(button='right')
            self.right_held = False
    
    def start_listeners(self, on_press=None, on_release=None):
        """Start keyboard listeners for manual override"""
        def handle_press(key):
            try:
                if on_press:
                    on_press(key.char if hasattr(key, 'char') else str(key))
            except AttributeError:
                pass
        
        def handle_release(key):
            try:
                if on_release:
                    on_release(key.char if hasattr(key, 'char') else str(key))
            except AttributeError:
                pass
        
        self.keyboard_listener = pynput.keyboard.Listener(
            on_press=handle_press,
            on_release=handle_release
        )
        self.keyboard_listener.start()
    
    def stop_listeners(self):
        """Stop all listeners"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None


if __name__ == "__main__":
    print("Input Controller Test")
    print("=" * 50)
    print("Testing human-like input simulation...")
    print("⚠️  ESC kill-switch is armed — press ESC to abort at any time.\n")
    
    controller = InputController()
    
    print("\nSimulating WASD movement...")
    for key in ['W', 'A', 'S', 'D']:
        controller.press_key(key, duration=0.3)
    
    print("\nSimulating mouse movement...")
    controller.move_mouse_relative(0.5, 0.0)
    time.sleep(0.2)
    controller.move_mouse_relative(-0.5, 0.0)
    
    print("\nSimulating click...")
    controller.click('left')
    
    print("\nTest complete!")
    print("Note: Run this only when ready for actual mouse/keyboard control")
