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
