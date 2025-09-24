#!/usr/bin/env python3
"""
Main entry point for Weapon Detection System
AISOLO Technologies Pvt. Ltd.

A computer vision system for security applications that can detect dangerous weapons 
in real-time surveillance footage.

Usage:
    python main.py detect --model models/weights/best.pt --source video.mp4 --save-video --save-log
    python main.py detect --model models/weights/best.pt --source 0 --show  # webcam
    python main.py train --data dataset/unified_weapon_dataset/data.yaml --epochs 50
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.weapon_detection.cli import detect_command, train_command, merge_command, download_command


def create_parser():
    """Create main argument parser."""
    parser = argparse.ArgumentParser(
        description="Weapon Detection System for Security Applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Detect weapons in video file
    python main.py detect --model models/weights/best.pt --source video.mp4 --save-video --save-log
    
    # Live webcam detection
    python main.py detect --model models/weights/best.pt --source 0 --show
    
    # Process RTSP stream
    python main.py detect --model models/weights/best.pt --source rtsp://camera_url --show
    
    # Train new model
    python main.py train --data dataset/unified_weapon_dataset/data.yaml --epochs 50
    
    # Merge datasets
    python main.py merge --config config/merge_config.yaml --output dataset/unified_weapon_dataset
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Detection command
    detect_parser = subparsers.add_parser('detect', help='Detect weapons in video/images')
    detect_command.add_args(detect_parser)
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train weapon detection model')
    train_command.add_args(train_parser)
    
    # Dataset merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple datasets')
    merge_command.add_args(merge_parser)
    
    # Dataset download command
    download_parser = subparsers.add_parser('download', help='Download COCO no-weapon dataset')
    download_command.add_args(download_parser)
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command
    if args.command == 'detect':
        return detect_command.main(args)
    elif args.command == 'train':
        return train_command.main(args)
    elif args.command == 'merge':
        return merge_command.main(args)
    elif args.command == 'download':
        return download_command.main(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())