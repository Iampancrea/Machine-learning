"""
Main entry point for Roblox Sword Fight Bot
Provides CLI interface for all bot operations
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Roblox Sword Fight Bot - ML-powered gameplay agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py record --duration 1800          # Record 30 minutes of gameplay
  python main.py train_bc --epochs 50            # Train behavior cloning model
  python main.py run --model checkpoints/model.pt  # Run trained bot
  python main.py train_rl --episodes 1000        # Fine-tune with RL
        """
    )
    
    parser.add_argument('command', 
                       choices=['record', 'train_bc', 'train_rl', 'run', 'test'],
                       help='Command to execute')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (for run command)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (for train_bc)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of RL episodes (for train_rl)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--duration', type=int, default=1800,
                       help='Recording duration in seconds')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='Device for training/inference')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ROBLOX SWORD FIGHT BOT")
    print("=" * 60)
    print(f"Command: {args.command}")
    print(f"Config: {args.config}")
    print("=" * 60)
    
    # Execute command
    if args.command == 'record':
        run_recording(args)
    elif args.command == 'train_bc':
        run_behavior_cloning(args)
    elif args.command == 'train_rl':
        run_reinforcement_learning(args)
    elif args.command == 'run':
        run_bot(args)
    elif args.command == 'test':
        run_tests(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


def run_recording(args):
    """Run data collection/recording with live input capture"""
    print("\n📹 Starting Data Recording...")
    print(f"Duration: {args.duration} seconds ({args.duration/60:.1f} minutes)")
    
    try:
        from data_collection.recorder import DataRecorder, ActionLogger
        from utils.config import load_config
        from utils.input_control import _start_esc_kill_switch
        import pynput.keyboard
        import pynput.mouse
        
        # Arm ESC kill-switch
        _start_esc_kill_switch()
        
        config = load_config(args.config.replace('configs/', ''))
        recorder = DataRecorder(config=config.config)
        action_logger = ActionLogger()
        
        # Setup pynput listeners to capture actual inputs
        def on_key_press(key):
            try:
                action_logger.press_key(key.char if hasattr(key, 'char') and key.char else str(key))
            except AttributeError:
                pass
        
        def on_key_release(key):
            try:
                action_logger.release_key(key.char if hasattr(key, 'char') and key.char else str(key))
            except AttributeError:
                pass
        
        def on_click(x, y, button, pressed):
            if pressed:
                if button == pynput.mouse.Button.left:
                    action_logger.click('left')
                elif button == pynput.mouse.Button.right:
                    action_logger.click('right')
        
        def on_move(x, y):
            # Track relative movement via delta from last position
            if hasattr(on_move, 'last_x'):
                dx = (x - on_move.last_x) / 800.0  # Normalize
                dy = (y - on_move.last_y) / 600.0
                action_logger.move_mouse(dx, dy)
            on_move.last_x = x
            on_move.last_y = y
        
        kb_listener = pynput.keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        mouse_listener = pynput.mouse.Listener(on_click=on_click, on_move=on_move)
        kb_listener.start()
        mouse_listener.start()
        
        print("\n⚠️  IMPORTANT: Make sure Roblox is running and visible!")
        print("🔑  Press ESC to hard-kill at any time")
        print("🎮  Your keyboard + mouse inputs are being recorded!\n")
        
        session_id = recorder.start_session()
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < args.duration:
            elapsed = time.time() - start_time
            progress = (elapsed / args.duration) * 100
            
            # Get actual player inputs from pynput listeners
            action = action_logger.get_action()
            recorder.record_frame(action)
            
            if int(elapsed) % 10 == 0:
                stats = recorder.get_statistics()
                print(f"\rProgress: {progress:.1f}% | Frames: {stats.get('total_frames', 0)} | "
                      f"FPS: {stats.get('avg_fps', 0):.1f} | "
                      f"Enemies: {stats.get('enemy_detection_rate', 0)*100:.0f}% | "
                      f"Safe: {stats.get('safe_zone_rate', 0)*100:.0f}%",
                      end='', flush=True)
            
            time.sleep(1/30)  # ~30 FPS
        
        # Cleanup listeners
        kb_listener.stop()
        mouse_listener.stop()
        
        recorder.stop_session(save=True)
        
        print("\n\n✅ Recording complete!")
        stats = recorder.get_statistics()
        print(f"Total frames: {stats.get('total_frames', 0)}")
        print(f"Enemy detection rate: {stats.get('enemy_detection_rate', 0)*100:.1f}%")
        print(f"Safe zone rate: {stats.get('safe_zone_rate', 0)*100:.1f}%")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Recording interrupted by user")
        if 'kb_listener' in locals():
            kb_listener.stop()
        if 'mouse_listener' in locals():
            mouse_listener.stop()
        if 'recorder' in locals():
            recorder.stop_session(save=True)
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_behavior_cloning(args):
    """Train behavior cloning model"""
    print("\n🤖 Training Behavior Cloning Model...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    try:
        from training.train_bc import BehaviorCloningTrainer
        from utils.config import load_config
        
        config = load_config(args.config.replace('configs/', ''))
        trainer = BehaviorCloningTrainer(config=config.config)
        
        print("\nLoading training data...")
        # Trainer will load data from ./data/recordings
        
        print(f"\nStarting training on {config['hardware']['device']}...")
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print("\n✅ Training complete!")
        print(f"Model saved to: {config['paths']['model_dir']}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_reinforcement_learning(args):
    """Train with reinforcement learning"""
    print("\n🎮 Starting Reinforcement Learning...")
    print(f"Episodes: {args.episodes}")
    
    try:
        from training.train_rl import RLTrainer
        from utils.config import load_config
        
        config = load_config(args.config.replace('configs/', ''))
        trainer = RLTrainer(config=config.config)
        
        print(f"\nStarting PPO training on {config['hardware']['device']}...")
        trainer.train(episodes=args.episodes)
        
        print("\n✅ RL training complete!")
        
    except Exception as e:
        print(f"\n❌ Error during RL training: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_bot(args):
    """Run the trained bot"""
    print("\n🚀 Running Trained Bot...")
    
    if not args.model:
        print("❌ Error: --model argument required for run command")
        sys.exit(1)
    
    try:
        from inference.bot_controller import BotController
        from utils.config import load_config
        
        config = load_config(args.config.replace('configs/', ''))
        controller = BotController(model_path=args.model, config=config.config)
        
        print("\n⚠️  IMPORTANT: Make sure Roblox is running!")
        print("Press ESC to hard-kill the bot at any time\n")
        
        controller.run()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error running bot: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_tests(args):
    """Run component tests"""
    print("\n🧪 Running Component Tests...\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Configuration loading
    print("Test 1: Configuration loading...")
    try:
        from utils.config import load_config
        config = load_config()
        print("  ✅ Config loaded successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        tests_failed += 1
    
    # Test 2: Network creation
    print("Test 2: Neural network creation...")
    try:
        from models.network import MLPNetwork
        model = MLPNetwork(input_dim=30, output_dim=8)
        print(f"  ✅ Network created ({model.num_params:,} parameters)")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Network test failed: {e}")
        tests_failed += 1
    
    # Test 3: Feature extraction
    print("Test 3: Feature extraction...")
    try:
        import numpy as np
        from feature_extraction.feature_engineer import FeatureEngineer
        engineer = FeatureEngineer()
        frame = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
        features = engineer.extract_features(frame, enemy_box=(100, 50, 40, 80))
        print(f"  ✅ Features extracted (shape: {features.shape})")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Feature test failed: {e}")
        tests_failed += 1
    
    # Summary
    print(f"\n{'='*40}")
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print(f"{'='*40}")
    
    if tests_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
