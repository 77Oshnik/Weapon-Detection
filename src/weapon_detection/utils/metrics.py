"""Metrics and evaluation utilities."""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from enum import Enum

class ThreatLevel(Enum):
    """Threat level enumeration."""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    
    def __lt__(self, other):
        return self.value < other.value
    
    def __le__(self, other):
        return self.value <= other.value
    
    def __gt__(self, other):
        return self.value > other.value
    
    def __ge__(self, other):
        return self.value >= other.value


def calculate_metrics(detection_log: List[Dict], total_frames: int) -> Dict[str, Any]:
    """Calculate detection metrics.
    
    Args:
        detection_log: List of detection events
        total_frames: Total number of processed frames
        
    Returns:
        Dictionary of metrics
    """
    if not detection_log:
        return {
            'total_frames': total_frames,
            'total_detections': 0,
            'detection_rate': 0.0,
            'class_distribution': {},
            'confidence_stats': {}
        }
    
    # Class distribution
    class_counts = {}
    confidences_by_class = {}
    
    for detection in detection_log:
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Count by class
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Collect confidences
        if class_name not in confidences_by_class:
            confidences_by_class[class_name] = []
        confidences_by_class[class_name].append(confidence)
    
    # Calculate confidence statistics
    confidence_stats = {}
    for class_name, confs in confidences_by_class.items():
        confidence_stats[class_name] = {
            'mean': sum(confs) / len(confs),
            'min': min(confs),
            'max': max(confs),
            'count': len(confs)
        }
    
    return {
        'total_frames': total_frames,
        'total_detections': len(detection_log),
        'detection_rate': len(detection_log) / max(total_frames, 1),
        'class_distribution': class_counts,
        'confidence_stats': confidence_stats,
        'unique_frames_with_detections': len(set(d['frame_number'] for d in detection_log))
    }


def generate_report(metrics: Dict[str, Any], output_path: Path = None) -> str:
    """Generate a formatted detection report.
    
    Args:
        metrics: Metrics dictionary
        output_path: Optional path to save report
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 50,
        "WEAPON DETECTION REPORT",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SUMMARY:",
        f"  Total Frames Processed: {metrics['total_frames']:,}",
        f"  Total Weapon Detections: {metrics['total_detections']:,}",
        f"  Detection Rate: {metrics['detection_rate']:.3f}",
        f"  Frames with Detections: {metrics.get('unique_frames_with_detections', 0):,}",
        ""
    ]
    
    # Class distribution
    if metrics['class_distribution']:
        report_lines.extend([
            "CLASS DISTRIBUTION:",
            ""
        ])
        
        for class_name, count in metrics['class_distribution'].items():
            percentage = (count / metrics['total_detections']) * 100
            report_lines.append(f"  {class_name.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        report_lines.append("")
    
    # Confidence statistics
    if metrics['confidence_stats']:
        report_lines.extend([
            "CONFIDENCE STATISTICS:",
            ""
        ])
        
        for class_name, stats in metrics['confidence_stats'].items():
            report_lines.extend([
                f"  {class_name.capitalize()}:",
                f"    Mean Confidence: {stats['mean']:.3f}",
                f"    Min Confidence: {stats['min']:.3f}",
                f"    Max Confidence: {stats['max']:.3f}",
                f"    Total Detections: {stats['count']:,}",
                ""
            ])
    
    report_lines.append("=" * 50)
    
    report_text = "\n".join(report_lines)
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding='utf-8')
    
    return report_text


def save_detection_log(detection_log: List[Dict], metrics: Dict[str, Any], 
                      output_path: Path, processing_time: float = None):
    """Save detection log and metrics to JSON file.
    
    Args:
        detection_log: List of detection events
        metrics: Metrics dictionary
        output_path: Path to save JSON file
        processing_time: Optional processing time in seconds
    """
    output_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'version': '1.0.0'
        },
        'metrics': metrics,
        'detections': detection_log
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, default=str)


# Export ThreatLevel for convenience
__all__ = ['ThreatLevel', 'calculate_metrics', 'generate_report', 'save_detection_log']