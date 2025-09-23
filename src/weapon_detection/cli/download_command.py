"""COCO no-weapon dataset download command."""

import random
import sys
from pathlib import Path

# Import the download functionality from scripts (temporary)
# In a full refactor, this would be moved to data module
try:
    sys.path.append(str(Path(__file__).parent.parent.parent.parent / "scripts"))
    from download_no_weapon_coco import main as download_main
except ImportError:
    download_main = None


def add_args(parser):
    """Add download arguments to parser."""
    parser.add_argument("--split", type=str, default="train2017", 
                        choices=["train2017", "val2017"],
                        help="COCO split to sample from")
    parser.add_argument("--output", type=Path, default=Path("dataset/no_weapon_coco"),
                        help="Output base directory")
    parser.add_argument("--max-images", type=int, default=1000,
                        help="Max number of images to download")
    parser.add_argument("--blacklist", type=str, default="knife",
                        help="Comma-separated category names to exclude")
    parser.add_argument("--person-only", action="store_true",
                        help="Require 'person' present in the image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8, 
                        help="Parallel download workers")


def main(args):
    """Execute download command."""
    if download_main is None:
        print(f"[ERROR] Download functionality not available")
        print(f"[INFO] Please run: python scripts/download_no_weapon_coco.py instead")
        return 1
    
    # Convert args to sys.argv format for compatibility
    argv_backup = sys.argv
    try:
        sys.argv = [
            'download_no_weapon_coco.py',
            '--split', args.split,
            '--outdir', str(args.output),
            '--max-images', str(args.max_images),
            '--blacklist', args.blacklist,
            '--seed', str(args.seed),
            '--train-ratio', str(args.train_ratio),
            '--val-ratio', str(args.val_ratio),
            '--test-ratio', str(args.test_ratio),
            '--workers', str(args.workers)
        ]
        
        if args.person_only:
            sys.argv.append('--person-only')
        
        return download_main()
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return 1
    finally:
        sys.argv = argv_backup