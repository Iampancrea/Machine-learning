# Machine-learning
Making Ai play roblox

## Overview
This project implements an autonomous agent for classic Roblox sword fighting games using a hybrid approach of Behavior Cloning (Imitation Learning) and Reinforcement Learning, optimized for systems with integrated graphics (Intel Iris Xe).

## Architecture

### Why This Approach Works for Your Hardware:
- **Feature-based input** instead of raw pixels reduces compute requirements by 95%+
- **Lightweight models** (~100K-500K parameters) train efficiently on CPU/iGPU
- **Modular design** allows incremental training without catastrophic forgetting

### Pipeline Stages:

1. **Data Collection Module**
   - Records your gameplay (screen + inputs)
   - Extracts features: enemy position, distance, health, sword cooldown
   - Stores state-action pairs for behavior cloning

2. **Feature Extraction**
   - Color-based detection for player/enemy identification
   - Template matching for UI elements (health bars, cooldowns)
   - Simple OCR for numerical values (optional, CPU-intensive)
   - Reduces 1920x1080 pixels → ~20 meaningful features

3. **Behavior Cloning (Supervised Learning)**
   - Trains MLP/CNN to mimic human actions
   - Input: Game features + recent action history
   - Output: Keyboard/mouse commands
   - Fast training on CPU (<30 min per epoch)

4. **Reinforcement Learning Fine-tuning**
   - Reward function: damage dealt, damage taken, kill/death ratio
   - PPO or DQN algorithm (lightweight implementations)
   - Trains in custom Roblox server or against bots

5. **Action Execution**
   - PyAutoGUI for keyboard/mouse control
   - Configurable reaction time delays (human-like behavior)
   - Safety mechanisms to prevent detection

## Project Structure

```
roblox_sword_bot/
├── data_collection/
│   ├── recorder.py          # Records gameplay + inputs
│   └── dataset.py           # Manages training datasets
├── feature_extraction/
│   ├── screen_processor.py  # Captures & processes frames
│   ├── color_detector.py    # Finds players/enemies by color
│   ├── template_matcher.py  # Detects UI elements
│   └── feature_engineer.py  # Combines all features
├── models/
│   ├── behavior_cloning.py  # Supervised learning model
│   ├── rl_agent.py          # RL fine-tuning agent
│   └── network.py           # Neural network architecture
├── training/
│   ├── train_bc.py          # Behavior cloning trainer
│   ├── train_rl.py          # RL trainer
│   └── rewards.py           # Reward function definitions
├── inference/
│   └── bot_controller.py    # Runs the trained bot
├── utils/
│   ├── input_control.py     # Keyboard/mouse automation
│   ├── config.py            # Configuration settings
│   └── logger.py            # Logging utilities
├── configs/
│   └── default_config.yaml  # All hyperparameters
├── requirements.txt         # Python dependencies
├── main.py                  # Entry point
└── README.md               # This file
```

## Hardware Optimization Strategies

### For Intel Iris Xe:
1. **Model Size**: Keep networks under 1M parameters
2. **Batch Size**: Use small batches (16-32) to fit in memory
3. **Precision**: Use FP16 where possible for 2x speedup
4. **Frame Rate**: Process every 2nd-3rd frame (30→10 FPS)
5. **Resolution**: Downscale captures to 320x180 or 160x90

### Expected Performance:
- **Inference**: 50-100 FPS on CPU (more than enough for 60 FPS game)
- **Training BC**: 1-2 hours per epoch on CPU
- **Training RL**: 4-8 hours for basic competency

## Getting Started

### Prerequisites:
- Python 3.8-3.11
- Roblox installed and logged in
- Classic sword fighting game selected

### Installation:
```bash
pip install -r requirements.txt
```

### Quick Start:
1. **Collect Data** (30-60 minutes of your gameplay):
   ```bash
   python main.py record --game "classic_sword" --duration 1800
   ```

2. **Train Behavior Cloning Model**:
   ```bash
   python main.py train_bc --epochs 50 --batch_size 32
   ```

3. **Test the Bot**:
   ```bash
   python main.py run --model checkpoints/bc_model.pt
   ```

4. **Fine-tune with RL** (optional, advanced):
   ```bash
   python main.py train_rl --episodes 1000
   ```

## Key Features

- ✅ Optimized for low-end hardware (no dedicated GPU required)
- ✅ Modular architecture for easy scaling
- ✅ Human-like reaction times (configurable delays)
- ✅ Anti-detection features (randomized timing)
- ✅ Comprehensive logging and monitoring
- ✅ Easy configuration via YAML files

## Next Steps for Scaling

Once working on classic sword fights:
1. Add more complex feature extraction (weapon types, multiple enemies)
2. Implement multi-agent scenarios
3. Add voice chat response (advanced)
4. Scale to other game modes (team battles, FFA)

## Warnings & Ethics

⚠️ **Important**: 
- Using bots may violate Roblox Terms of Service
- Risk of account suspension or banning
- Use only on alternate accounts for experimentation
- This is for educational purposes only

## License
MIT License - Educational use only

## Contributing
Feel free to submit issues and enhancement requests!
