"""Training command implementation."""

import json
import time
from pathlib import Path
from ultralytics import YOLO


def add_args(parser):
    """Add training arguments to parser."""
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


def main(args):
    """Execute training command."""
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
    
    # Add optional parameters
    if args.device:
        train_params['device'] = args.device
    if args.resume:
        train_params['resume'] = args.resume
    
    # Print configuration
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
        print(f"[INFO] Best weights: {results.save_dir}/weights/best.pt")
        print(f"[INFO] Results: {results.save_dir}")
        
        # Save training summary
        summary = {
            'model': args.model,
            'dataset': args.data,
            'epochs': args.epochs,
            'batch_size': args.batch,
            'training_time_seconds': training_time,
            'best_weights': str(results.save_dir / 'weights' / 'best.pt'),
            'results_dir': str(results.save_dir)
        }
        
        summary_path = Path(results.save_dir) / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[INFO] Training summary: {summary_path}")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return 1