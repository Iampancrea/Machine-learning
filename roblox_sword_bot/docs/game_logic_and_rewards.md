# Roblox Sword Fight: Game Logic & AI Analysis

This document serves as the foundational "Game Sense" for the Reinforcement Learning agent. It outlines the absolute truths of the classic Roblox Sword Fight mechanics and how our bot perceives and acts upon them given its hardware constraints (Intel Iris Xe).

## 1. Core Mechanics (The "Classic" Rules)
- **The Objective:** Survive and accumulate kills/points. 
- **The Weapon:** The classic Roblox sword (equipped on slot `1`). It requires timing. Spamming the mouse button does not equal faster kills; there is an internal swing cooldown/animation lock. The tip of the sword must connect with the enemy's hitbox.
- **Movement (Footwork):** Sword fighting in Roblox is 90% footwork. Players must constantly strafe (`A`, `D`), move in for a strike (`W`), and immediately backpedal (`S`) to avoid the counter-swing. Jumping (`SPACE`) is frequently used to make the hitbox unpredictable.
- **Health System:** Players start with 100 HP.
  - Enemy health is visible via bright green `100 HP` text above their heads.
  - Our health is only visible when we take damage (the top right pill-shaped Roblox health bar appears and shifts from Green -> Yellow -> Red as it depletes).
- **The Safe Zone:** A designated area where combat is disabled. Entering it provides safety, but killing a player outside it steals 500 points. Camping here is detrimental to winning.
- **Kill Confirmation:** The game broadcasts kills in the bottom right corner (e.g., `PlayerA stole 500 🕐 from PlayerB 7 studs away`).

## 2. AI Perception (How the Bot "Sees" the World)
Since we cannot read memory (anti-cheat restrictions), the bot relies entirely on hyper-optimized computer vision.
- **The Arena:** The bot captures the full `1536x888` screen.
- **Enemy Detection:** It scans for the specific HSV signature of the green `100 HP` text above heads. It masks out the dead center of the screen to avoid mistaking its own health text for an enemy.
- **Spatial Awareness (CNN):** The full screen is downscaled to a tiny `80x60` grayscale image. A second "Mask" channel is overlaid with bright white dots exactly where enemy heads are located, giving the bot crystal-clear spatial coordinates.
- **Temporal Memory:** The bot remembers the last 10 frames of enemy positions (Frame Stacking) and whether it just swung its sword (Action Memory). This simulates an LSTM/RNN without the massive CPU bottleneck.
- **Self-Awareness:** It constantly monitors the top right corner. If the health bar appears, it calculates its exact HP percentage by measuring the bar's width.

## 3. The Dense Reward Shaping Strategy
Sparse rewards (e.g., +10 for a kill, -10 for a death) are too rare for the bot to learn quickly. We must implement **Dense Rewards**—micro-rewards for doing "the right things" that lead to a kill.

Based on the game logic, these are the micro-behaviors we must reward/penalize:
1. **Distance Closing:** Reward the bot for decreasing the distance to the nearest enemy.
2. **Target Tracking:** Reward the bot for keeping the enemy near the center of the screen (crosshairs).
3. **Engaging (Swinging):** Reward the bot for clicking the mouse *only* when an enemy is within striking distance (close to the center). Penalize swinging at nothing (wasting cooldowns).
4. **Self-Preservation:** Heavily penalize taking damage (when `player_health` decreases frame-over-frame).

This document is the absolute ground truth for the RL environment tuning moving forward.

## 4. Newly Discovered Game Logic & AI Lessons Learned

During active reinforcement learning iterations, we discovered several critical nuances in Roblox's game logic and UI interactions, requiring structural changes in the perception and control layers:

### A. Boundary Fluctuations (Safe Zone Symmetries)
* **The Issue:** The safe zone boundary is not clean; visual noise or subtle character movements cause the detector to flicker rapidly between `Safe: True` and `Safe: False`.
* **Asymmetry Trap:** If leaving the zone gives a reward of `+2.0` only once, but re-entering penalizes the bot `-2.0` every time, a flickering boundary creates an infinite negative reward loop (cowardice trap).
* **The Solution:** We balanced the state machine. The initial leave gives `+2.0`, subsequent re-entries penalize `-2.0`, but leaving the zone *again* awards a counter-balancing `+2.0`. This ensures boundary fluctuations remain net-neutral while still enforcing a per-frame camping penalty of `-0.05` for actually staying inside.

### B. Intrusive UI Popups (Bank & Follow Us Panels)
* **The Issue:** The game dynamically spawns popup windows (such as the Bank Menu and the "Follow Us" promotional panel). These popups capture mouse and keyboard focus, preventing the bot from looking around or fighting.
* **The Solution:** 
  1. We added OpenCV template matching for both the Bank UI and the Follow Us panel's red close buttons (`bank_x_template.png` and `follow_x_template.png`).
  2. The bot is penalized `-5.0` for letting a popup stay open.
  3. **Control Breakout:** To click the 'X' button, the bot must temporarily break out of Shift Lock (by tapping `SHIFT`), wait for Roblox to release mouse lock, perform a `force_click` on the screen coordinates of the button, wait, and tap `SHIFT` again to re-engage combat camera lock.

### C. Respawn & Inventory Loading Latency
* **The Issue:** When a player dies and respawns, Roblox does not load the player's tools/inventory instantly. Pressing the slot key `1` immediately on spawn often fails to equip the sword, leaving the bot weaponless.
* **The Solution:** We increased the post-respawn sleep to `2.0` seconds to allow the game to settle, and programmed the controller to double-tap `1` (with a `0.1` second delay) to guarantee the sword is successfully equipped.

### D. Strict Episode Truncation
* **The Issue:** Because the environment runs continuously, if the bot is too passive or fails to die, the training loop would play infinitely in a single episode, ignoring the configured `steps_per_episode: 500` limit.
* **The Solution:** We implemented an explicit check at step 500 to set `truncated = True`, forcing a hard reset of the environment to start a fresh episode.

