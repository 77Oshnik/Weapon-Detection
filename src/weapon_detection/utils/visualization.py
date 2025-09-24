"""Visualization utilities for weapon detection."""

import cv2
import numpy as np
from typing import List, Dict
from .metrics import ThreatLevel


# Color schemes
CLASS_COLORS = {
    'pistol': (0, 0, 255),      # Red
    'rifle': (0, 0, 255),       # Red  
    'knife': (0, 165, 255),     # Orange
    'no_weapon': (0, 255, 0)    # Green
}

THREAT_COLORS = {
    ThreatLevel.SAFE: (0, 255, 0),      # Green
    ThreatLevel.LOW: (0, 255, 255),     # Yellow
    ThreatLevel.MEDIUM: (0, 165, 255),  # Orange
    ThreatLevel.HIGH: (0, 0, 255)       # Red
}


def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw detection boxes and labels on frame.
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries
        
    Returns:
        Frame with drawn detections
    """
    annotated = frame.copy()
    
    for det in detections:
        class_name = det['class_name']
        conf = det['confidence']
        x1, y1, x2, y2 = [int(x) for x in det['bbox']]
        
        # Get color for this class
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return annotated


def create_threat_overlay(frame: np.ndarray, threat_level: ThreatLevel) -> np.ndarray:
    """Create threat level overlay on frame.
    
    Args:
        frame: Input frame
        threat_level: Current threat level
        
    Returns:
        Frame with threat overlay
    """
    annotated = frame.copy()
    
    # Get threat color and name
    threat_names = {
        ThreatLevel.SAFE: "SAFE",
        ThreatLevel.LOW: "LOW", 
        ThreatLevel.MEDIUM: "MEDIUM",
        ThreatLevel.HIGH: "HIGH"
    }
    
    color = THREAT_COLORS.get(threat_level, (255, 255, 255))
    threat_name = threat_names.get(threat_level, "UNKNOWN")
    
    # Draw threat level indicator
    cv2.rectangle(annotated, (10, 10), (200, 50), color, -1)
    cv2.putText(annotated, f"THREAT: {threat_name}", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return annotated


def create_detection_summary(detections: List[Dict]) -> str:
    """Create a text summary of detections.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Summary string
    """
    if not detections:
        return "No weapons detected"
    
    weapon_detections = [d for d in detections if d['class_name'] in ['pistol', 'rifle', 'knife']]
    
    if not weapon_detections:
        return "No weapons detected"
    
    # Count by class
    class_counts = {}
    for det in weapon_detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Create summary
    parts = []
    for class_name, count in class_counts.items():
        parts.append(f"{count} {class_name}{'s' if count > 1 else ''}")
    
    return f"Detected: {', '.join(parts)}"