"""
GameDetector — Full game-state detection suite for Roblox sword-fight bot.

Detection pipeline:
    1. detect_enemies      — HSV green "HP" text detection + contour clustering
    2. detect_safe_zone    — static ROI color check for safe zone banner
    3. detect_kill_log     — OCR on kill log region (bottom-right of full screen)
    4. detect_death        — health bar greying detection (top-right of full screen)

Frame assumptions:
    • 800×600 RGB input for detect_enemies/detect_safe_zone (centered capture)
    • Full screen (1536×888) grabs via mss for kill_log and death detection
"""

import numpy as np
import cv2
import mss
import time
from typing import List, Tuple, Optional
from collections import deque

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class GameDetector:
    """Detect enemies, kills, deaths, and game-state elements in a Roblox
    sword-fight game using HSV color segmentation, contour analysis, and
    targeted OCR."""

    def __init__(self, config: dict = None):
        """
        Args:
            config: Full config dict (the whole YAML).
                    Unrecognised keys are silently ignored.
        """
        cfg = config or {}
        features_cfg = cfg.get('features', {})

        # ── Health-bar HSV ranges (legacy, kept for compatibility) ────
        self.green_lower = np.array(cfg.get("green_lower", [35, 80, 80]))
        self.green_upper = np.array(cfg.get("green_upper", [85, 255, 255]))
        self.red_lower1 = np.array(cfg.get("red_lower1", [0, 80, 80]))
        self.red_upper1 = np.array(cfg.get("red_upper1", [10, 255, 255]))
        self.red_lower2 = np.array(cfg.get("red_lower2", [170, 80, 80]))
        self.red_upper2 = np.array(cfg.get("red_upper2", [180, 255, 255]))

        # ── Geometric constraints (legacy) ───────────────────────────
        self.min_bar_width: int = cfg.get("min_bar_width", 20)
        self.max_bar_width: int = cfg.get("max_bar_width", 150)
        self.max_bar_height: int = cfg.get("max_bar_height", 12)
        self.min_aspect_ratio: float = cfg.get("min_aspect_ratio", 3.0)

        # ── Safe-zone banner ROI (relative to 800×600 frame) ────────
        sz_cfg = features_cfg.get('safe_zone', {})
        self.safe_zone_roi: Tuple[int, int, int, int] = tuple(
            sz_cfg.get("roi", (150, 5, 500, 45))
        )
        self.banner_red_lower1 = np.array(sz_cfg.get("banner_red_lower_1", [0, 120, 150]))
        self.banner_red_upper1 = np.array(sz_cfg.get("banner_red_upper_1", [10, 255, 255]))
        self.banner_red_lower2 = np.array(sz_cfg.get("banner_red_lower_2", [170, 120, 150]))
        self.banner_red_upper2 = np.array(sz_cfg.get("banner_red_upper_2", [180, 255, 255]))
        self.banner_yellow_lower = np.array(sz_cfg.get("banner_yellow_lower", [15, 120, 150]))
        self.banner_yellow_upper = np.array(sz_cfg.get("banner_yellow_upper", [35, 255, 255]))

        # ── White-text confirmation thresholds (legacy) ──────────────
        self.white_threshold: int = cfg.get("white_threshold", 200)
        self.text_confirm_ratio: float = cfg.get("text_confirm_ratio", 0.03)

        # ── Player name ──────────────────────────────────────────────
        kl_cfg = features_cfg.get('kill_log', {})
        self.player_name = kl_cfg.get("player_name", "sagupaam6").lower()

        # ── Kill Log OCR Setup ───────────────────────────────────────
        self.use_ocr = features_cfg.get("use_ocr", False)
        self.reader = None
        if self.use_ocr:
            if EASYOCR_AVAILABLE:
                print("🔤 Initializing EasyOCR for Kill Log tracking...")
                use_gpu = cfg.get("hardware", {}).get("use_gpu", False)
                self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
                print("✅ EasyOCR ready.")
            else:
                print("⚠️  easyocr not installed! Disabling OCR. pip install easyocr")
                self.use_ocr = False

        # Kill log screen region (absolute coordinates on the full screen)
        self.kill_log_region = kl_cfg.get("screen_region", [350, 795, 1186, 93])
        self.kill_log_confidence = kl_cfg.get("confidence_threshold", 0.4)
        self.kill_log_dedup_timeout = kl_cfg.get("dedup_timeout_seconds", 4.0)

        # Deduplication: track recently seen kill/death events
        # Each entry is (event_text_hash, timestamp)
        self._recent_events: deque = deque(maxlen=20)

        # ── Death Detection Setup ────────────────────────────────────
        dd_cfg = features_cfg.get('death_detection', {})
        self.death_detection_enabled = dd_cfg.get("enabled", True)
        self.death_bar_region = dd_cfg.get("screen_region", [1698, 20, 222, 45])
        self.death_template_path = dd_cfg.get("template_path", "checkpoints/death_bar_template.png")
        self.death_match_threshold = dd_cfg.get("match_threshold", 0.80)
        self.death_template = None
        if self.death_detection_enabled:
            import os
            if os.path.exists(self.death_template_path):
                self.death_template = cv2.imread(self.death_template_path)
                print(f"💀 Loaded death bar template from: {self.death_template_path}")
            else:
                print(f"⚠️  Death bar template not found at {self.death_template_path}! Disabling template-based death detection.")
                self.death_detection_enabled = False

        # ── Enemy Detection Setup ────────────────────────────────────
        ed_cfg = features_cfg.get('enemy_detection', {})
        self.enemy_detection_enabled = ed_cfg.get("enabled", True)
        self.hp_hsv_lower = np.array(ed_cfg.get("hp_text_hsv_lower", [35, 80, 150]))
        self.hp_hsv_upper = np.array(ed_cfg.get("hp_text_hsv_upper", [85, 255, 255]))
        self.ed_min_area = ed_cfg.get("min_contour_area", 15)
        self.ed_max_area = ed_cfg.get("max_contour_area", 8000)
        self.ed_min_aspect = ed_cfg.get("min_aspect_ratio", 1.2)
        self.ed_max_aspect = ed_cfg.get("max_aspect_ratio", 20.0)
        self.ed_exclude_bottom = ed_cfg.get("exclude_bottom_ratio", 0.35)
        self.ed_exclude_top = ed_cfg.get("exclude_top_ratio", 0.05)
        self.ed_cluster_dist = ed_cfg.get("cluster_distance_px", 50)

        # mss instance for separate screen captures (kill log, health bar)
        self._sct = mss.MSS()

        # ── Player Name Template Setup ───────────────────────────────
        self.player_template_path = "checkpoints/player_name_template.png"
        self.player_template = None
        if os.path.exists(self.player_template_path):
            tmp = cv2.imread(self.player_template_path, cv2.IMREAD_GRAYSCALE)
            _, self.player_template = cv2.threshold(tmp, 200, 255, cv2.THRESH_BINARY)
            print("👤 Loaded and thresholded player name template for perfect self-exclusion.")

        # ── Bank UI Template Setup ───────────────────────────────────
        self.bank_template_path = "checkpoints/bank_x_template.png"
        self.bank_template = None
        if os.path.exists(self.bank_template_path):
            tmp_bank = cv2.imread(self.bank_template_path, cv2.IMREAD_COLOR)
            self.bank_template = cv2.cvtColor(tmp_bank, cv2.COLOR_BGR2RGB)
            print("🏦 Loaded Bank UI Red 'X' template.")

    def detect_bank_ui(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detects if the massive Bank UI is open by template matching the Red 'X' button.
        Returns the (x, y) coordinates of the center of the Red 'X' if found, else None.
        """
        if self.bank_template is None:
            return None
            
        res = cv2.matchTemplate(frame, self.bank_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # 0.8 is a solid threshold for identical UI elements
        if max_val > 0.8:
            h, w = self.bank_template.shape[:2]
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return (center_x, center_y)
            
        return None

    # ─────────────────────────────────────────────────────────────────
    # Kill Log OCR (grabs its own ROI from the full screen)
    # ─────────────────────────────────────────────────────────────────
    def detect_kill_log(self, frame: np.ndarray = None) -> dict:
        """
        Detect kill/death events by reading the kill log at the bottom-right
        of the FULL SCREEN (separate from the 800x600 game capture).

        Kill log format:
            "[killer] stole [amount] 🕐 from [victim] [N] studs away"

        Returns:
            dict with keys:
                'kill'   (bool) — we killed someone
                'death'  (bool) — someone killed us
                'killer' (str)  — who killed us
                'victim' (str)  — who we killed
        """
        default_result = {'kill': False, 'death': False, 'killer': '', 'victim': ''}

        if not self.use_ocr or self.reader is None:
            return default_result

        try:
            # Grab kill log ROI from absolute screen coordinates
            region = self.kill_log_region
            monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3],
            }
            screenshot = self._sct.grab(monitor)
            roi = np.array(screenshot)[:, :, :3]  # Drop alpha (BGRA → BGR)

            # Pre-process: the kill log has white/colored text on a semi-dark bg
            # Convert to grayscale and threshold to isolate bright text
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

            # Run EasyOCR on the small pre-processed crop (MUCH faster than full frame)
            allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- "
            results = self.reader.readtext(thresh, detail=0, paragraph=True, allowlist=allowed_chars)
            text_full = " ".join(results).lower().strip()

        except Exception as e:
            return default_result

        if not text_full:
            return default_result

        result = {'kill': False, 'death': False, 'killer': '', 'victim': ''}
        player = self.player_name

        # Generate a simple hash of the text for deduplication
        text_hash = hash(text_full)
        now = time.time()

        # Purge old events
        while self._recent_events and (now - self._recent_events[0][1]) > self.kill_log_dedup_timeout:
            self._recent_events.popleft()

        # Check if we already processed this exact text recently
        recent_hashes = {h for h, _ in self._recent_events}
        if text_hash in recent_hashes:
            return default_result

        # ── Parse kill events ────────────────────────────────────────
        # "sagupaam6 stole 500 from [victim]"
        if f"{player} stole" in text_full or f"{player} st" in text_full:
            result['kill'] = True
            try:
                after_stole = text_full.split(f"{player} stole" if f"{player} stole" in text_full else f"{player} st")[1]
                if "from " in after_stole:
                    victim_raw = after_stole.split("from ")[1].strip()
                    # Victim name is the first word after "from"
                    result['victim'] = victim_raw.split()[0] if victim_raw.split() else 'someone'
                else:
                    result['victim'] = 'someone'
            except (IndexError, ValueError):
                result['victim'] = 'someone'

        # ── Parse death events ───────────────────────────────────────
        # "[killer] stole 500 from sagupaam6"
        if f"from {player}" in text_full or f"from {player[:6]}" in text_full:
            result['death'] = True
            try:
                # Find the text before "stole" to get killer name
                match_str = f"from {player}" if f"from {player}" in text_full else f"from {player[:6]}"
                lines = text_full.split(match_str)
                if lines[0]:
                    before_from = lines[0]
                    if "stole" in before_from or "st" in before_from:
                        split_word = "stole" if "stole" in before_from else "st"
                        killer_part = before_from.split(split_word)[0].strip()
                        words = killer_part.split()
                        result['killer'] = words[-1] if words else 'someone'
                    else:
                        words = before_from.strip().split()
                        result['killer'] = words[-1] if words else 'someone'
                else:
                    result['killer'] = 'someone'
            except (IndexError, ValueError):
                result['killer'] = 'someone'

        # Store event for deduplication
        if result['kill'] or result['death']:
            self._recent_events.append((text_hash, now))
            print(f"    [OCR] Kill log text: '{text_full}'")

        return result

    # ─────────────────────────────────────────────────────────────────
    # Death Detection (health bar greying out in top-right corner)
    # ─────────────────────────────────────────────────────────────────
    def detect_death(self) -> bool:
        """
        Detect if the player is dead by template matching the greyed-out health
        bar in the top-right corner.

        Returns:
            True if the player is dead.
        """
        if not self.death_detection_enabled or self.death_template is None:
            return False

        try:
            region = self.death_bar_region
            monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3],
            }
            screenshot = self._sct.grab(monitor)
            roi = np.array(screenshot)[:, :, :3]  # BGRA → BGR

            # Match template
            res = cv2.matchTemplate(roi, self.death_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            return max_val > self.death_match_threshold

        except Exception:
            return False

    def get_player_health(self) -> float:
        """
        Estimates the player's health percentage by looking at the top-right health bar.
        Returns 1.0 if the bar is hidden (full health), or a float between 0.0 and 1.0.
        """
        if not self.death_detection_enabled:
            return 1.0
            
        try:
            region = self.death_bar_region
            monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3],
            }
            screenshot = self._sct.grab(monitor)
            roi = np.array(screenshot)[:, :, :3]  # BGRA → BGR
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Color ranges for the health bar
            mask_g = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
            mask_y = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
            mask_r1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask_r2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
            
            mask_total = mask_g | mask_y | mask_r1 | mask_r2
            
            # Find columns that have at least one colored pixel
            colored_cols = np.where(np.any(mask_total > 0, axis=0))[0]
            
            if len(colored_cols) == 0:
                # If no health bar colors are detected, the bar is hidden (health is 100%)
                return 1.0
                
            bar_width = colored_cols[-1] - colored_cols[0]
            hp_pct = float(bar_width) / (region[2] - 4) # subtracting 4 for borders
            
            return max(0.01, min(1.0, hp_pct)) # min 1% if visible
            
        except Exception:
            return 1.0

    # ─────────────────────────────────────────────────────────────────
    # Enemy Detection (green HP text nametags)
    # ─────────────────────────────────────────────────────────────────
    def detect_enemies(self, frame: np.ndarray) -> List[dict]:
        """
        Detect enemy players by finding the bright green "XX HP" text
        floating above their heads.

        Uses HSV color filtering → contour detection → spatial filtering
        → clustering to identify individual enemies.

        Args:
            frame: RGB uint8 image (800×600 centered capture)

        Returns:
            List of enemy dicts, each with:
                'player_center': (x, y) — estimated player position
                'tag_center':    (x, y) — center of HP text
                'hp_pct':        float  — always 1.0 (can't read exact HP)
                'hp_bar':        tuple  — bounding rect of the HP text
                'text_confidence': float — 0.0 (no OCR)
        """
        if not self.enemy_detection_enabled:
            return []

        height, width = frame.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Mask for HP text colors (Green, Yellow, and Red depending on health levels)
        green_mask = cv2.inRange(hsv, self.hp_hsv_lower, self.hp_hsv_upper)
        yellow_mask = cv2.inRange(hsv, np.array([15, 80, 150]), np.array([35, 255, 255]))
        red_mask_hp1 = cv2.inRange(hsv, np.array([0, 80, 150]), np.array([15, 255, 255]))
        red_mask_hp2 = cv2.inRange(hsv, np.array([165, 80, 150]), np.array([180, 255, 255]))
        red_mask_hp = cv2.bitwise_or(red_mask_hp1, red_mask_hp2)
        
        hp_mask = cv2.bitwise_or(green_mask, yellow_mask)
        hp_mask = cv2.bitwise_or(hp_mask, red_mask_hp)
        
        # Mask for the distinct BRIGHT Red Clock icon above every player's head
        r_mask1 = cv2.inRange(hsv, np.array([0, 150, 200]), np.array([10, 255, 255]))
        r_mask2 = cv2.inRange(hsv, np.array([170, 150, 200]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(r_mask1, r_mask2)

        # Spatial filtering: zero out bottom (green floor) and top (banner)
        exclude_top_px = int(height * self.ed_exclude_top)
        exclude_bottom_px = int(height * (1.0 - self.ed_exclude_bottom))
        hp_mask[:exclude_top_px, :] = 0
        hp_mask[exclude_bottom_px:, :] = 0

        # Exclude leftmost ~20% (UI elements like Kills counter, score)
        ui_cutoff = int(width * 0.18)
        hp_mask[:, :ui_cutoff] = 0

        # Morphological operations to clean up noise and connect text fragments
        kernel = np.ones((self.ed_min_area, self.ed_min_area), np.uint8)
        hp_mask = cv2.morphologyEx(hp_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(hp_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)



        # Filter contours by size, shape, and aspect ratio
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.ed_min_area or area > self.ed_max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            
            # The "100 HP" text is small and horizontal.
            # Filter out vertical swords (h > 30) and long slanted lines (w > 120 or h > 40)
            if h < 4 or h > 30 or w < 10 or w > 120:
                continue
                
            cx = x + w // 2
            cy = y + h // 2

            # ── Red Clock Verification (Foolproof enemy confirmation) ──
            # The red timer clock is ALWAYS slightly left and above the "100 HP" text.
            roi_y1 = max(0, y - 35)
            roi_y2 = max(0, y)
            roi_x1 = max(0, x - 40)
            roi_x2 = min(width, x + 30)
            
            red_pixels = cv2.countNonZero(red_mask[roi_y1:roi_y2, roi_x1:roi_x2])
            if red_pixels < 8:  # Need at least a few red pixels to confirm it's a real player's clock
                continue

            aspect = w / h
            if aspect < self.ed_min_aspect or aspect > self.ed_max_aspect:
                continue

            cx = x + w // 2
            cy = y + h // 2
            candidates.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area,
            })

        if not candidates:
            return []

        # ── Player Exclusion Logic (Scale-Invariant & Zero Deadzones) ──
        # Since the camera is behind the player, the player's own nametag is ALWAYS 
        # the lowest candidate (maximum Y coordinate) in the center vertical column.
        center_candidates = [c for c in candidates if abs(c['center'][0] - (width // 2)) < 60]
        if center_candidates:
            player_cand = max(center_candidates, key=lambda c: c['center'][1])
            candidates.remove(player_cand)

        if not candidates:
            return []

        # Cluster nearby candidates into individual enemies
        enemies = self._cluster_detections(candidates, frame.shape)

        return enemies

    def _cluster_detections(self, candidates: List[dict],
                             frame_shape: tuple) -> List[dict]:
        """
        Cluster nearby green text detections into individual enemies.
        Multiple green blobs close together likely belong to the same nametag.
        """
        if not candidates:
            return []

        height, width = frame_shape[:2]
        max_dist = self.ed_cluster_dist

        # Simple greedy clustering
        used = set()
        clusters = []

        # Sort by area (largest first — more likely to be actual HP text)
        candidates.sort(key=lambda c: c['area'], reverse=True)

        for i, cand in enumerate(candidates):
            if i in used:
                continue

            cluster = [cand]
            used.add(i)

            cx, cy = cand['center']

            for j, other in enumerate(candidates):
                if j in used:
                    continue
                ox, oy = other['center']
                dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                if dist < max_dist:
                    cluster.append(other)
                    used.add(j)

            clusters.append(cluster)

        # Convert clusters to enemy dicts
        enemies = []
        for cluster in clusters:
            # Average center of all blobs in this cluster
            avg_x = int(np.mean([c['center'][0] for c in cluster]))
            avg_y = int(np.mean([c['center'][1] for c in cluster]))

            # Bounding rect of the largest blob
            main = max(cluster, key=lambda c: c['area'])
            bbox = main['bbox']

            # Player center is estimated ~50px below the HP text
            player_y = min(avg_y + 50, height - 1)

            enemies.append({
                'player_center': (avg_x, player_y),
                'tag_center': (avg_x, avg_y),
                'hp_pct': 1.0,  # Can't read exact HP without OCR
                'hp_bar': bbox,
                'text_confidence': 0.0,
            })

        return enemies

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
    print(f"║  OCR enabled     : {det.use_ocr}")
    print(f"║  Player name     : {det.player_name}")
    print(f"║  Kill log region : {det.kill_log_region}")
    print(f"║  Death bar region: {det.death_bar_region}")
    print(f"║  Enemy detection : {det.enemy_detection_enabled}")
    print(f"║  HP HSV lower    : {det.hp_hsv_lower.tolist()}")
    print(f"║  HP HSV upper    : {det.hp_hsv_upper.tolist()}")
    print(f"║  Safe-zone ROI   : x={det.safe_zone_roi[0]}, y={det.safe_zone_roi[1]}, "
          f"w={det.safe_zone_roi[2]}, h={det.safe_zone_roi[3]}")
    print("╚══════════════════════════════════════════════╝")
    print("\nColorDetector alias active:", ColorDetector is GameDetector)
