"""
YOLOv8-nano training script for Roblox Sword Fight Bot.
Use this script to train your own enemy detector using ultralytics.
"""
import os
import argparse
import sys
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: ultralytics is not installed. Run 'pip install ultralytics'")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-nano on Roblox screenshots")
    parser.add_argument('--data', type=str, required=True, help="Path to your dataset YAML (e.g. data.yaml)")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--img_size', type=int, default=640, help="Image size")
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help="Initial weights (yolov8n.pt for nano)")
    
    args = parser.parse_args()
    
    print(f"🚀 Initializing YOLOv8-nano training with {args.weights}...")
    
    # Load model
    model = YOLO(args.weights)
    
    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        device=0, # Assuming GPU
        project='runs/yolo',
        name='roblox_enemy_detector'
    )
    
    print("\n✅ Training complete!")
    print(f"Best weights saved to: {results.save_dir}/weights/best.pt")
    print("Update your default_config.yaml 'yolo_model_path' to point to these weights.")

if __name__ == '__main__':
    main()
