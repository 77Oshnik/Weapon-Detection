"""Dataset merge command implementation."""

from pathlib import Path
from ..data.dataset_merger import DatasetMerger


def add_args(parser):
    """Add merge arguments to parser."""
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for unified dataset")
    parser.add_argument("--config", type=Path, 
                        default=Path("config/merge_config.yaml"),
                        help="Configuration file for dataset merging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without copying files")


def main(args):
    """Execute merge command."""
    try:
        merger = DatasetMerger(args.config)
        stats = merger.merge(args.output, args.dry_run)
        
        if not args.dry_run:
            print(f"[SUCCESS] Dataset merge completed successfully")
            print(f"[INFO] Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Dataset merge failed: {e}")
        return 1