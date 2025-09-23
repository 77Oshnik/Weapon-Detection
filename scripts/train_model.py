#!/usr/bin/env python3
"""
Train YOLOv8 model on unified weapon detection dataset.
Supports multiple model sizes, hyperparameter tuning, and comprehensive logging.

Usage:
  python scripts/train_model.py --data dataset/unified_weapon_dataset/data.yaml --model yolov8n.pt --epochs 50 --batch 16
"""

import argparse
import json
import time
from pathlib import Path
from ultralytics import YOLO
import yaml


def load_config(config_path: Path = None) -> dict:
    """Load training configuration from YAML file."""
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for weapon detection")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset YAML file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                        help="YOLOv8 model variant")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Project directory")
    parser.add_argument("--name", type=str, default="weapon_detection",
                        help="Experiment name")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Weight decay")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, default="",
                        help="Device to train on (e.g., '0' for GPU, 'cpu')")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of data loader workers")
    parser.add_argument("--save-period", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # Load model
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Training parameters
    train_params = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'project': args.project,
        'name': args.name,
        'patience': args.patience,
        'lr0': args.lr0,
        'weight_decay': args.weight_decay,
        'workers': args.workers,
        'save_period': args.save_period,
        'verbose': True,
        'plots': True,
        'save_json': True,
    }
    
    # Add device if specified
    if args.device:
        train_params['device'] = args.device
    
    # Add resume if specified
    if args.resume:
        train_params['resume'] = args.resume
    
    # Print training configuration
    print(f"[INFO] Training configuration:")
    for key, value in train_params.items():
        print(f"  {key}: {value}")
    
    # Start training
    print(f"[INFO] Starting training...")
    start_time = time.time()
    
    try:
        results = model.train(**train_params)
        
        training_time = time.time() - start_time
        print(f"[SUCCESS] Training completed in {training_time:.2f} seconds")
        print(f"[INFO] Best weights saved to: {results.save_dir}/weights/best.pt")
        print(f"[INFO] Last weights saved to: {results.save_dir}/weights/last.pt")
        print(f"[INFO] Results saved to: {results.save_dir}")
        
        # Save training summary
        summary = {
            'model': args.model,
            'dataset': args.data, 
            'epochs': args.epochs,
            'batch_size': args.batch,
            'image_size': args.imgsz,
            'training_time_seconds': training_time,
            'best_weights': str(results.save_dir / 'weights' / 'best.pt'),
            'last_weights': str(results.save_dir / 'weights' / 'last.pt'),
            'results_dir': str(results.save_dir)
        }
        
        summary_path = Path(results.save_dir) / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[INFO] Training summary saved to: {summary_path}")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())