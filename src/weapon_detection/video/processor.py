"""Video processing for weapon detection."""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional, Callable, Any
from pathlib import Path
import time
from datetime import datetime

from ..detector import WeaponDetector
from ..utils.metrics import ThreatLevel


class VideoProcessor:
    """Process video files and streams for weapon detection."""
    
    def __init__(self, detector: WeaponDetector, output_path: Optional[str] = None,
                 save_detections: bool = True, fps: Optional[int] = None):
        """Initialize video processor.
        
        Args:
            detector: WeaponDetector instance
            output_path: Path to save output video
            save_detections: Whether to save detection results
            fps: Output video FPS (defaults to input fps)
        """
        self.detector = detector
        self.output_path = output_path
        self.save_detections = save_detections
        self.fps = fps
        self.video_writer = None
        
        # Processing stats
        self.start_time = None
        self.frames_processed = 0
        self.total_frames = 0
        
    def process_video(self, video_path: str, show: bool = False, 
                     progress_callback: Optional[Callable] = None) -> dict:
        """Process video file for weapon detection.
        
        Args:
            video_path: Path to input video
            show: Whether to show video during processing
            progress_callback: Function to call with progress updates
            
        Returns:
            Processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_fps = self.fps or original_fps
        
        print(f"[INFO] Processing video: {video_path}")
        print(f"[INFO] Resolution: {width}x{height}, FPS: {original_fps}, Frames: {self.total_frames}")
        
        # Setup video writer
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, output_fps, (width, height)
            )
        
        # Process frames
        self.start_time = time.time()
        frame_number = 0
        threat_frames = 0
        max_threat_level = ThreatLevel.SAFE
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect weapons
                timestamp = time.time()
                detections, threat_level = self.detector.detect(
                    frame, frame_number, timestamp
                )
                
                # Track highest threat level
                if threat_level.value > max_threat_level.value:
                    max_threat_level = threat_level
                
                if threat_level != ThreatLevel.SAFE:
                    threat_frames += 1
                
                # Annotate frame
                if self.save_detections or show or self.output_path:
                    annotated_frame = self.detector.annotate_frame(
                        frame, detections, threat_level
                    )
                else:
                    annotated_frame = frame
                
                # Save frame
                if self.video_writer:
                    self.video_writer.write(annotated_frame)
                
                # Show frame
                if show:
                    cv2.imshow('Weapon Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] Processing stopped by user")
                        break
                
                frame_number += 1
                self.frames_processed = frame_number
                
                # Progress callback
                if progress_callback and frame_number % 30 == 0:
                    progress = frame_number / self.total_frames
                    progress_callback(progress, frame_number, threat_frames)
                
        finally:
            cap.release()
            if self.video_writer:
                self.video_writer.release()
            if show:
                cv2.destroyAllWindows()
        
        # Calculate statistics
        processing_time = time.time() - self.start_time
        avg_fps = self.frames_processed / processing_time if processing_time > 0 else 0
        
        stats = {
            'frames_processed': self.frames_processed,
            'total_frames': self.total_frames,
            'processing_time': processing_time,
            'avg_fps': avg_fps,
            'threat_frames': threat_frames,
            'threat_percentage': (threat_frames / max(self.frames_processed, 1)) * 100,
            'max_threat_level': max_threat_level.name,
            'detector_stats': self.detector.get_statistics()
        }
        
        print(f"\n[SUMMARY] Video processing completed")
        print(f"[INFO] Processed {self.frames_processed}/{self.total_frames} frames")
        print(f"[INFO] Processing time: {processing_time:.2f}s ({avg_fps:.2f} FPS)")
        print(f"[INFO] Threat frames: {threat_frames} ({stats['threat_percentage']:.1f}%)")
        print(f"[INFO] Max threat level: {max_threat_level.name}")
        
        return stats
    
    def process_stream(self, source: Any = 0, show: bool = True) -> Generator[Tuple[np.ndarray, list, str], None, None]:
        """Process live video stream.
        
        Args:
            source: Video source (0 for webcam, rtsp url, etc.)
            show: Whether to show video window
            
        Yields:
            Tuple of (frame, detections, threat_level)
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        print(f"[INFO] Starting live stream processing from {source}")
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Failed to read frame from stream")
                    break
                
                # Detect weapons
                timestamp = time.time()
                detections, threat_level = self.detector.detect(
                    frame, frame_number, timestamp
                )
                
                # Annotate frame
                annotated_frame = self.detector.annotate_frame(
                    frame, detections, threat_level
                )
                
                # Show frame
                if show:
                    cv2.imshow('Live Weapon Detection', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] Stream stopped by user")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = f"detection_screenshot_{timestamp_str}.jpg"
                        cv2.imwrite(screenshot_path, annotated_frame)
                        print(f"[INFO] Screenshot saved: {screenshot_path}")
                
                frame_number += 1
                yield annotated_frame, detections, threat_level
                
        finally:
            cap.release()
            if show:
                cv2.destroyAllWindows()
    
    def extract_threat_clips(self, video_path: str, output_dir: str, 
                           clip_duration: float = 10.0, 
                           min_threat_level: ThreatLevel = ThreatLevel.MEDIUM) -> list:
        """Extract video clips containing weapon detections.
        
        Args:
            video_path: Input video path
            output_dir: Directory to save clips
            clip_duration: Duration of each clip in seconds
            min_threat_level: Minimum threat level to extract
            
        Returns:
            List of extracted clip paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # First pass: identify threat moments
        threat_moments = []
        frame_number = 0
        
        print(f"[INFO] Analyzing video for threat moments...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections, threat_level = self.detector.detect(frame, frame_number)
            
            if threat_level.value >= min_threat_level.value:
                timestamp = frame_number / fps
                threat_moments.append({
                    'frame': frame_number,
                    'timestamp': timestamp,
                    'threat_level': threat_level,
                    'detections': detections
                })
            
            frame_number += 1
        
        cap.release()
        
        if not threat_moments:
            print("[INFO] No threat moments found")
            return []
        
        # Merge nearby moments and extract clips
        clips = []
        clip_duration_frames = int(clip_duration * fps)
        
        i = 0
        while i < len(threat_moments):
            start_frame = max(0, threat_moments[i]['frame'] - clip_duration_frames // 2)
            end_frame = threat_moments[i]['frame'] + clip_duration_frames // 2
            
            # Merge overlapping clips
            while (i + 1 < len(threat_moments) and 
                   threat_moments[i + 1]['frame'] <= end_frame):
                i += 1
                end_frame = max(end_frame, threat_moments[i]['frame'] + clip_duration_frames // 2)
            
            # Extract clip
            clip_path = output_dir / f"threat_clip_{start_frame:06d}_{end_frame:06d}.mp4"
            if self._extract_clip(video_path, clip_path, start_frame, end_frame, fps):
                clips.append(str(clip_path))
                print(f"[INFO] Extracted clip: {clip_path}")
            
            i += 1
        
        return clips
    
    def _extract_clip(self, video_path: str, output_path: Path, 
                     start_frame: int, end_frame: int, fps: float) -> bool:
        """Extract a video clip between specified frames."""
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        try:
            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to extract clip: {e}")
            return False
            
        finally:
            cap.release()
            writer.release()