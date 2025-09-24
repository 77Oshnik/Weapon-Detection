"""Model training utilities for weapon detection."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from ultralytics import YOLO
import yaml


class ModelTrainer:
    """Model training wrapper for YOLOv8 weapon detection."""
    
    def __init__(self, model_name: str = "yolov8n.pt", project_dir: str = "runs/train"):
        """Initialize model trainer.
        
        Args:
            model_name: YOLOv8 model variant
            project_dir: Project directory for training outputs
        """
        self.model_name = model_name
        self.project_dir = Path(project_dir)
        self.model = None
        self.training_results = None
        
    def train(self, data_config: str, epochs: int = 50, batch_size: int = 16,
              imgsz: int = 640, **kwargs) -> Dict[str, Any]:
        """Train the weapon detection model.
        
        Args:
            data_config: Path to data.yaml file
            epochs: Number of training epochs
            batch_size: Training batch size
            imgsz: Input image size
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        print(f"[INFO] Initializing model: {self.model_name}")
        self.model = YOLO(self.model_name)
        
        # Prepare training parameters
        train_params = {
            'data': data_config,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'project': str(self.project_dir),
            'name': kwargs.get('name', 'weapon_detection'),
            'patience': kwargs.get('patience', 20),
            'lr0': kwargs.get('lr0', 0.01),
            'weight_decay': kwargs.get('weight_decay', 0.0005),
            'workers': kwargs.get('workers', 8),
            'save_period': kwargs.get('save_period', 10),
            'verbose': True,
            'plots': True,
            'save_json': True,
        }
        
        # Add device if specified
        if 'device' in kwargs and kwargs['device']:
            train_params['device'] = kwargs['device']
        
        # Add resume if specified
        if 'resume' in kwargs and kwargs['resume']:
            train_params['resume'] = kwargs['resume']
        
        print(f"[INFO] Training parameters:")
        for key, value in train_params.items():
            print(f"  {key}: {value}")
        
        # Start training
        print(f"[INFO] Starting training...")
        start_time = time.time()
        
        try:
            self.training_results = self.model.train(**train_params)
            
            training_time = time.time() - start_time
            
            # Prepare results summary
            results = {
                'model': self.model_name,
                'dataset': data_config,
                'epochs': epochs,
                'batch_size': batch_size,
                'training_time_seconds': training_time,
                'best_weights': str(self.training_results.save_dir / 'weights' / 'best.pt'),
                'last_weights': str(self.training_results.save_dir / 'weights' / 'last.pt'),
                'results_dir': str(self.training_results.save_dir),
                'success': True
            }
            
            print(f"[SUCCESS] Training completed in {training_time:.2f} seconds")
            print(f"[INFO] Best weights: {results['best_weights']}")
            print(f"[INFO] Results directory: {results['results_dir']}")
            
            # Save training summary
            self._save_training_summary(results)
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': self.model_name,
                'dataset': data_config
            }
    
    def evaluate(self, data_config: str, model_path: str, **kwargs) -> Dict[str, Any]:
        """Evaluate trained model on test set.
        
        Args:
            data_config: Path to data.yaml file
            model_path: Path to trained model weights
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation results
        """
        print(f"[INFO] Loading model for evaluation: {model_path}")
        model = YOLO(model_path)
        
        # Run validation
        val_params = {
            'data': data_config,
            'split': kwargs.get('split', 'test'),
            'save_json': True,
            'plots': True
        }
        
        if 'device' in kwargs and kwargs['device']:
            val_params['device'] = kwargs['device']
        
        try:
            results = model.val(**val_params)
            
            eval_results = {
                'model_path': model_path,
                'dataset': data_config,
                'map50': float(results.box.map50),
                'map50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'success': True
            }
            
            print(f"[INFO] Evaluation results:")
            print(f"  mAP@0.5: {eval_results['map50']:.4f}")
            print(f"  mAP@0.5:0.95: {eval_results['map50_95']:.4f}")
            print(f"  Precision: {eval_results['precision']:.4f}")
            print(f"  Recall: {eval_results['recall']:.4f}")
            
            return eval_results
            
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_path': model_path
            }
    
    def _save_training_summary(self, results: Dict[str, Any]):
        """Save training summary to JSON file."""
        try:
            summary_path = Path(results['results_dir']) / 'training_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"[INFO] Training summary saved: {summary_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save training summary: {e}")
    
    def get_best_model_path(self) -> Optional[str]:
        """Get path to best trained model weights."""
        if self.training_results:
            return str(self.training_results.save_dir / 'weights' / 'best.pt')
        return None
    
    def load_data_config(self, data_path: str) -> Dict[str, Any]:
        """Load and validate data configuration.
        
        Args:
            data_path: Path to data.yaml file
            
        Returns:
            Data configuration dictionary
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = ['train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data_config:
                    raise ValueError(f"Missing required field in data.yaml: {field}")
            
            print(f"[INFO] Data configuration loaded: {data_path}")
            print(f"  Classes: {data_config['nc']}")
            print(f"  Names: {data_config['names']}")
            
            return data_config
            
        except Exception as e:
            print(f"[ERROR] Failed to load data config: {e}")
            raise


class TrainingScheduler:
    """Training scheduler for automated model training."""
    
    def __init__(self, base_config: Dict[str, Any]):
        """Initialize training scheduler.
        
        Args:
            base_config: Base training configuration
        """
        self.base_config = base_config
        self.experiments = []
    
    def add_experiment(self, name: str, config_overrides: Dict[str, Any]):
        """Add training experiment to schedule.
        
        Args:
            name: Experiment name
            config_overrides: Configuration overrides for this experiment
        """
        experiment = {
            'name': name,
            'config': {**self.base_config, **config_overrides}
        }
        self.experiments.append(experiment)
        print(f"[INFO] Added experiment: {name}")
    
    def run_experiments(self) -> List[Dict[str, Any]]:
        """Run all scheduled experiments.
        
        Returns:
            List of experiment results
        """
        results = []
        
        for i, experiment in enumerate(self.experiments, 1):
            print(f"\n[INFO] Running experiment {i}/{len(self.experiments)}: {experiment['name']}")
            
            trainer = ModelTrainer(
                model_name=experiment['config'].get('model', 'yolov8n.pt'),
                project_dir=experiment['config'].get('project', 'runs/train')
            )
            
            # Update experiment config with name
            config = experiment['config'].copy()
            config['name'] = experiment['name']
            
            # Run training
            result = trainer.train(**config)
            result['experiment_name'] = experiment['name']
            results.append(result)
            
            if result['success']:
                print(f"[SUCCESS] Experiment {experiment['name']} completed")
            else:
                print(f"[ERROR] Experiment {experiment['name']} failed: {result.get('error', 'Unknown error')}")
        
        print(f"\n[SUMMARY] Completed {len(results)} experiments")
        successful = sum(1 for r in results if r['success'])
        print(f"  Successful: {successful}/{len(results)}")
        
        return results