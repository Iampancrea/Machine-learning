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

    # ── Player Health & Kill Log ─────────────────────────────────────
    
    def detect_player_health(self, frame: np.ndarray) -> float:
        """
        Check the player's own health bar at the top right of the screen.
        Returns a value from 0.0 to 1.0 representing estimated health.
        """
        # Standard Roblox health bar ROI (approximate top right)
        # Using a wide strip to catch it regardless of screen variations
        h, w = frame.shape[:2]
        roi = frame[10:40, int(w * 0.75):w - 10]
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Look for Green, Yellow, and Red pixels
        green = cv2.inRange(hsv, self.green_lower, self.green_upper)
        yellow = cv2.inRange(hsv, self.banner_yellow_lower, self.banner_yellow_upper)
        red1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        
        # Combine all health bar colors
        health_mask = cv2.bitwise_or(green, yellow)
        health_mask = cv2.bitwise_or(health_mask, red1)
        health_mask = cv2.bitwise_or(health_mask, red2)
        
        # Calculate ratio of health pixels in the ROI
        # Max expected pixels depends on ROI size, we normalize it
        health_pixels = cv2.countNonZero(health_mask)
        # Assume a full health bar occupies roughly 150x10 = 1500 pixels
        health_ratio = min(1.0, health_pixels / 1500.0)
        return health_ratio

    def detect_kill_log(self, frame: np.ndarray) -> bool:
        """
        Detect the kill log message at the bottom of the screen.
        Looks for the distinct red/yellow clock icon '⏰' in the bottom 100 pixels.
        Returns True if a kill log is present.
        """
        h, w = frame.shape[:2]
        roi = frame[h - 100:h, 0:w]
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # The clock icon has very distinct pure red and yellow pixels packed tightly
        red1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red = cv2.bitwise_or(red1, red2)
        yellow = cv2.inRange(hsv, self.banner_yellow_lower, self.banner_yellow_upper)
        
        # Dilate to connect the red and yellow parts of the clock
        kernel = np.ones((3, 3), np.uint8)
        red_dilated = cv2.dilate(red, kernel, iterations=1)
        yellow_dilated = cv2.dilate(yellow, kernel, iterations=1)
        
        # Intersection: Where red and yellow are right next to each other
        clock_mask = cv2.bitwise_and(red_dilated, yellow_dilated)
        
        # If we find a cluster of overlapping red/yellow, it's the clock icon
        return cv2.countNonZero(clock_mask) > 5

    # ─────────────────────────────────────────────────────────────────
    # STEP 1 — find health bars
    # ─────────────────────────────────────────────────────────────────
    def detect_health_bars(self, frame: np.ndarray) -> List[dict]:
        """Detect health-bar candidates via HSV masking + geometric filter.

        Args:
            frame: RGB uint8 image (H, W, 3).

        Returns:
            List of dicts, each with keys:
                'bbox'   — (x, y, w, h)
                'hp_pct' — float 0-1, estimated HP remaining
                'center' — (cx, cy)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Green mask
        mask_green = cv2.inRange(hsv, self.green_lower, self.green_upper)

        # Red mask (two hue ranges merged)
        mask_red1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask_red2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Combined health-bar mask
        mask = cv2.bitwise_or(mask_green, mask_red)

        # Morphology — horizontal kernel to bridge bar fragments, then
        # a small open to nuke isolated noise.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 2))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bars: List[dict] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Geometric gate
            if h < 2:
                continue
            aspect = w / h
            if (
                aspect < self.min_aspect_ratio
                or w < self.min_bar_width
                or w > self.max_bar_width
                or h > self.max_bar_height
            ):
                continue

            # HP estimation: green pixels / total coloured pixels inside bbox
            roi_hsv = hsv[y : y + h, x : x + w]
            green_in_roi = cv2.inRange(roi_hsv, self.green_lower, self.green_upper)
            red1_in_roi = cv2.inRange(roi_hsv, self.red_lower1, self.red_upper1)
            red2_in_roi = cv2.inRange(roi_hsv, self.red_lower2, self.red_upper2)
            red_in_roi = cv2.bitwise_or(red1_in_roi, red2_in_roi)

            green_count = int(np.count_nonzero(green_in_roi))
            red_count = int(np.count_nonzero(red_in_roi))
            total = green_count + red_count
            hp_pct = green_count / total if total > 0 else 0.0

            cx = x + w // 2
            cy = y + h // 2

            bars.append({
                "bbox": (x, y, w, h),
                "hp_pct": hp_pct,
                "center": (cx, cy),
            })

        return bars

    # ─────────────────────────────────────────────────────────────────
    # STEP 2 — confirm enemies (white-text anchor)
    # ─────────────────────────────────────────────────────────────────
    def confirm_enemies(
        self, frame: np.ndarray, health_bars: List[dict]
    ) -> List[dict]:
        """Filter health-bar candidates by checking for white username text
        immediately below each bar.

        Args:
            frame:       RGB uint8 image.
            health_bars: Output of :meth:`detect_health_bars`.

        Returns:
            List of confirmed enemy dicts:
                'hp_bar'          — (x, y, w, h)
                'hp_pct'          — float 0-1
                'player_center'   — (px, py), estimated body centre
                'tag_center'      — (tx, ty), health-bar centre
                'text_confidence' — float, white-pixel ratio in text ROI
        """
        frame_h, frame_w = frame.shape[:2]
        confirmed: List[dict] = []

        for bar in health_bars:
            bx, by, bw, bh = bar["bbox"]

            # Text ROI: a strip just below the health bar, padded 15 px
            # on each side to catch the full username.
            tx1 = max(0, bx - 15)
            ty1 = by + bh
            tx2 = min(frame_w, bx + bw + 15)
            ty2 = min(frame_h, by + bh + 25)

            if ty1 >= ty2 or tx1 >= tx2:
                continue  # ROI fell off the frame edge

            text_roi = frame[ty1:ty2, tx1:tx2]

            # White-pixel mask: all three RGB channels above threshold
            white_mask = (
                (text_roi[:, :, 0] > self.white_threshold)
                & (text_roi[:, :, 1] > self.white_threshold)
                & (text_roi[:, :, 2] > self.white_threshold)
            )
            white_ratio = float(np.count_nonzero(white_mask)) / white_mask.size

            if white_ratio >= self.text_confirm_ratio:
                cx, cy = bar["center"]
                confirmed.append({
                    "hp_bar": bar["bbox"],
                    "hp_pct": bar["hp_pct"],
                    "player_center": (cx, cy + 50),
                    "tag_center": (cx, cy),
                    "text_confidence": white_ratio,
                })

        return confirmed

    # ─────────────────────────────────────────────────────────────────
    # Convenience wrappers
    # ─────────────────────────────────────────────────────────────────
    def detect_enemies(self, frame: np.ndarray) -> List[dict]:
        """Full pipeline: detect bars → confirm via text anchor.

        Args:
            frame: RGB uint8 image.

        Returns:
            List of confirmed enemy dicts (see :meth:`confirm_enemies`).
        """
        bars = self.detect_health_bars(frame)
        return self.confirm_enemies(frame, bars)

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
