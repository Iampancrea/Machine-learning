"""
GameDetector — HSV color segmentation + geometric anchoring for Roblox
sword-fight bot detection.  No OCR, no template matching, just raw pixel
math the way the silicon gods intended.

Detection pipeline:
    1. detect_health_bars  — find red/green HP bars via HSV masks + shape filter
    2. confirm_enemies     — anchor each bar to a player by requiring white
                             username text directly below it
    3. detect_safe_zone    — check a static ROI for the red/yellow banner

Frame assumptions:
    • 800×600 RGB input (numpy uint8 array, channel order RGB)
    • Health bars are thin horizontal rectangles floating above heads
    • Usernames are white text rendered immediately below the bar
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class GameDetector:
    """Detect enemies and game-state elements in a Roblox sword-fight
    game using HSV color segmentation and geometric filtering."""

    def __init__(self, config: dict = None):
        """
        Args:
            config: Optional dict to override any default thresholds.
                    Unrecognised keys are silently ignored.
        """
        cfg = config or {}

        # ── Health-bar HSV ranges ────────────────────────────────────
        # Green portion of the bar
        self.green_lower = np.array(cfg.get("green_lower", [35, 80, 80]))
        self.green_upper = np.array(cfg.get("green_upper", [85, 255, 255]))

        # Red wraps around the hue wheel → two ranges
        self.red_lower1 = np.array(cfg.get("red_lower1", [0, 80, 80]))
        self.red_upper1 = np.array(cfg.get("red_upper1", [10, 255, 255]))
        self.red_lower2 = np.array(cfg.get("red_lower2", [170, 80, 80]))
        self.red_upper2 = np.array(cfg.get("red_upper2", [180, 255, 255]))

        # ── Geometric constraints for a valid health bar ─────────────
        self.min_bar_width: int = cfg.get("min_bar_width", 20)
        self.max_bar_width: int = cfg.get("max_bar_width", 150)
        self.max_bar_height: int = cfg.get("max_bar_height", 12)
        self.min_aspect_ratio: float = cfg.get("min_aspect_ratio", 3.0)

        # ── Safe-zone banner ROI (relative to 800×600 frame) ────────
        self.safe_zone_roi: Tuple[int, int, int, int] = tuple(
            cfg.get("safe_zone_roi", (150, 5, 500, 45))
        )  # (x, y, w, h)

        # Banner HSV — red component
        self.banner_red_lower1 = np.array(cfg.get("banner_red_lower1", [0, 120, 150]))
        self.banner_red_upper1 = np.array(cfg.get("banner_red_upper1", [10, 255, 255]))
        self.banner_red_lower2 = np.array(cfg.get("banner_red_lower2", [170, 120, 150]))
        self.banner_red_upper2 = np.array(cfg.get("banner_red_upper2", [180, 255, 255]))

        # Banner HSV — yellow component
        self.banner_yellow_lower = np.array(cfg.get("banner_yellow_lower", [15, 120, 150]))
        self.banner_yellow_upper = np.array(cfg.get("banner_yellow_upper", [35, 255, 255]))

        # ── White-text confirmation thresholds ───────────────────────
        self.white_threshold: int = cfg.get("white_threshold", 200)
        self.text_confirm_ratio: float = cfg.get("text_confirm_ratio", 0.03)

        # ── OCR Reader for Kill Log ────────────────────────────────────
        self.use_ocr = cfg.get("features", {}).get("use_ocr", False)
        self.reader = None
        if self.use_ocr:
            if EASYOCR_AVAILABLE:
                print("Initializing EasyOCR for Kill Log tracking... (this might take a second)")
                use_gpu = cfg.get("hardware", {}).get("use_gpu", False)
                self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            else:
                print("WARNING: easyocr is enabled in config but not installed! Disabling OCR features.")
                self.use_ocr = False
                
        self.player_name = cfg.get("features", {}).get("kill_log", {}).get("player_name", "sagupaam6")

    # ── Kill Log OCR ─────────────────────────────────────────────────
    
    def detect_kill_log(self, frame: np.ndarray) -> dict:
        """
        Detect kill/death events by reading the kill log text at the bottom-right
        of the FULL SCREEN (not the 800x600 centered capture, which doesn't reach
        the bottom-right corner).
        
        Kill log format in this game:
            "[killer] stole [amount] [clock icon] from [victim] [distance] studs away"
        
        Returns:
            dict with keys 'kill' (bool), 'death' (bool), 'killer' (str), 'victim' (str)
        """
        import mss
        
        default_result = {'kill': False, 'death': False, 'killer': '', 'victim': ''}
        
        if not self.use_ocr or self.reader is None:
            return default_result
            
        try:
            with mss.mss() as sct:
                # Grab the bottom-right corner of the primary monitor
                # Kill log occupies roughly bottom 150px, right 60% of screen
                monitor = sct.monitors[1]  # Primary monitor
                screen_w = monitor["width"]
                screen_h = monitor["height"]
                
                kill_log_region = {
                    "left": monitor["left"] + int(screen_w * 0.4),
                    "top": monitor["top"] + screen_h - 150,
                    "width": int(screen_w * 0.6),
                    "height": 150
                }
                
                screenshot = sct.grab(kill_log_region)
                roi = np.array(screenshot)[:, :, :3]  # Drop alpha, keep BGR
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return default_result
        
        # Run OCR on the kill log region
        try:
            results = self.reader.readtext(roi, detail=0)
            text_full = " ".join(results).lower()
        except Exception:
            return default_result
        
        if not text_full.strip():
            return default_result
            
        player_name = self.player_name
        
        result = {'kill': False, 'death': False, 'killer': '', 'victim': ''}
        
        # Check if we killed someone: "sagupaam6 stole ... from [victim]"
        if f"{player_name} stole" in text_full:
            result['kill'] = True
            try:
                after_from = text_full.split(f"{player_name} stole")[1]
                if "from " in after_from:
                    victim = after_from.split("from ")[1].split(" ")[0]
                    result['victim'] = victim
            except (IndexError, ValueError):
                result['victim'] = 'someone'
            
        # Check if someone killed us: "[killer] stole ... from sagupaam6"
        if f"from {player_name}" in text_full:
            result['death'] = True
            try:
                # Find the line containing "from sagupaam6"
                for line in text_full.split("\n"):
                    if f"from {player_name}" in line and "stole" in line:
                        killer = line.split("stole")[0].strip().split()[-1] if line.split("stole")[0].strip() else 'someone'
                        result['killer'] = killer
                        break
                if not result['killer']:
                    result['killer'] = 'someone'
            except (IndexError, ValueError):
                result['killer'] = 'someone'
        
        if result['kill'] or result['death']:
            print(f"    [OCR] Read: '{text_full}'")
        
        return result

    # ─────────────────────────────────────────────────────────────────
    # Enemy Detection (Disabled)
    # ─────────────────────────────────────────────────────────────────
    def detect_enemies(self, frame: np.ndarray) -> List[dict]:
        """
        Health bars are disabled in this game, so we return an empty list.
        The bot will rely exclusively on the CNN spatial branch and OCR kill logs.
        """
        return []

    def get_nearest_enemy(
        self,
        frame: np.ndarray,
        screen_center: Tuple[int, int] = None,
    ) -> Optional[dict]:
        """Return the confirmed enemy closest to *screen_center*.

        Args:
            frame:         RGB uint8 image.
            screen_center: Reference point; defaults to frame centre.

        Returns:
            Enemy dict or ``None``.
        """
        enemies = self.detect_enemies(frame)
        if not enemies:
            return None

        if screen_center is None:
            screen_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        sx, sy = screen_center
        best = None
        best_dist = float("inf")
        for enemy in enemies:
            px, py = enemy["player_center"]
            dist = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = enemy

        return best

    # ─────────────────────────────────────────────────────────────────
    # Safe-zone detection
    # ─────────────────────────────────────────────────────────────────
    def detect_safe_zone(self, frame: np.ndarray) -> bool:
        """Check if the 'YOU ARE IN THE SAFE ZONE' banner is visible.

        Looks for concentrated red/yellow pixels inside a fixed ROI at
        the top-centre of the frame.

        Args:
            frame: RGB uint8 image (expected 800×600).

        Returns:
            ``True`` if the banner is detected (player is in safe zone).
        """
        rx, ry, rw, rh = self.safe_zone_roi
        roi = frame[ry : ry + rh, rx : rx + rw]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        # Red component
        red1 = cv2.inRange(roi_hsv, self.banner_red_lower1, self.banner_red_upper1)
        red2 = cv2.inRange(roi_hsv, self.banner_red_lower2, self.banner_red_upper2)
        red_mask = cv2.bitwise_or(red1, red2)

        # Yellow component
        yellow_mask = cv2.inRange(
            roi_hsv, self.banner_yellow_lower, self.banner_yellow_upper
        )

        banner_mask = cv2.bitwise_or(red_mask, yellow_mask)
        banner_ratio = float(np.count_nonzero(banner_mask)) / banner_mask.size

        return banner_ratio > 0.08


# ── Backward compatibility ──────────────────────────────────────────
ColorDetector = GameDetector


if __name__ == "__main__":
    det = GameDetector()
    print("╔══════════════════════════════════════════════╗")
    print("║        GameDetector — Configuration          ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  Green HSV       : {det.green_lower.tolist()} → {det.green_upper.tolist()}")
    print(f"║  Red HSV (low)   : {det.red_lower1.tolist()} → {det.red_upper1.tolist()}")
    print(f"║  Red HSV (high)  : {det.red_lower2.tolist()} → {det.red_upper2.tolist()}")
    print(f"║  Bar width       : {det.min_bar_width} – {det.max_bar_width} px")
    print(f"║  Bar max height  : {det.max_bar_height} px")
    print(f"║  Min aspect ratio: {det.min_aspect_ratio}")
    print(f"║  Safe-zone ROI   : x={det.safe_zone_roi[0]}, y={det.safe_zone_roi[1]}, "
          f"w={det.safe_zone_roi[2]}, h={det.safe_zone_roi[3]}")
    print(f"║  White threshold : {det.white_threshold}")
    print(f"║  Text confirm %  : {det.text_confirm_ratio * 100:.1f}%")
    print("╚══════════════════════════════════════════════╝")
    print("\nColorDetector alias active:", ColorDetector is GameDetector)
