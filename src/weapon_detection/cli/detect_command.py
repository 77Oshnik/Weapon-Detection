"""Detection command implementation."""

import cv2
import time
from pathlib import Path
from datetime import datetime

from ..detector import WeaponDetector
from ..utils.metrics import calculate_metrics, generate_report, save_detection_log, ThreatLevel


def add_args(parser):
    """Add detection arguments to parser."""
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained YOLOv8 model (.pt file)")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (file path, webcam index, or RTSP URL)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--device", type=str, default="",
                        help="Device to run inference on")
    parser.add_argument("--show", action="store_true",
                        help="Display video in real-time")
    parser.add_argument("--save-video", action="store_true",
                        help="Save annotated video")
    parser.add_argument("--save-log", action="store_true",
                        help="Save detection log to JSON")
    parser.add_argument("--fps-limit", type=int, default=30,
                        help="Limit processing FPS")


def main(args):
    """Execute detection command."""
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = WeaponDetector(args.model, args.conf, args.device)
    
    # Open video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        return 1
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.source != "0" else 0
    
    print(f"[INFO] Video: {width}x{height} @ {fps} FPS")
    if total_frames > 0:
        print(f"[INFO] Total frames: {total_frames}")
    
    # Setup video writer
    out_writer = None
    if args.save_video:
        output_path = output_dir / f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"[INFO] Saving video: {output_path}")
    
    # Processing loop
    frame_number = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            current_time = time.time()
            
            # Detect weapons
            detections, threat_level = detector.detect(frame, frame_number, current_time)
            
            # Annotate frame
            annotated_frame = detector.annotate_frame(frame, detections, threat_level)
            
            # Progress for video files
            if total_frames > 0 and frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"[INFO] Progress: {progress:.1f}% ({frame_number}/{total_frames})")
            
            # Alert for threats
            if threat_level != ThreatLevel.SAFE:
                threat_names = {
                    ThreatLevel.LOW: "LOW",
                    ThreatLevel.MEDIUM: "MEDIUM", 
                    ThreatLevel.HIGH: "HIGH"
                }
                threat_name = threat_names.get(threat_level, "UNKNOWN")
                print(f"[ALERT] Frame {frame_number}: {threat_name} threat!")
                weapon_dets = [d for d in detections if d['class_name'] in detector.weapon_classes]
                for det in weapon_dets:
                    print(f"  -> {det['class_name']} (conf: {det['confidence']:.3f})")
            
            # Save/display frame
            if out_writer:
                out_writer.write(annotated_frame)
            
            if args.show:
                cv2.imshow('Weapon Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # FPS limiting
            if args.fps_limit > 0:
                time.sleep(max(0, 1/args.fps_limit - (time.time() - current_time)))
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()
        
        # Statistics
        processing_time = time.time() - start_time
        stats = detector.get_statistics()
        
        print(f"\n[SUMMARY] Processing completed")
        print(f"  Frames: {stats['total_frames']}")
        print(f"  Weapon detections: {stats['weapon_detections']}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Average FPS: {stats['total_frames'] / processing_time:.2f}")
        
        # Save logs
        if args.save_log and detector.detection_log:
            log_path = output_dir / f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            metrics = calculate_metrics(detector.detection_log, stats['total_frames'])
            save_detection_log(detector.detection_log, metrics, log_path, processing_time)
            print(f"[INFO] Detection log: {log_path}")
            
            # Generate report
            report_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report = generate_report(metrics, report_path)
            print(f"[INFO] Report: {report_path}")
    
    return 0