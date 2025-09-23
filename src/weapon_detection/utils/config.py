"""Configuration management utilities."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def save_config(config: Dict[str, Any], output_path: Union[str, Path]):
    """Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif output_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {output_path.suffix}")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for weapon detection system.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'type': 'yolov8n.pt',
            'confidence_threshold': 0.25,
            'device': '',  # auto-detect
            'input_size': 640
        },
        'classes': {
            'names': ['pistol', 'rifle', 'knife', 'no_weapon'],
            'weapon_classes': ['pistol', 'rifle', 'knife']
        },
        'threat_levels': {
            'high_confidence': 0.7,
            'medium_confidence': 0.5,
            'knife_threshold': 0.6
        },
        'video': {
            'fps_limit': 30,
            'show_display': False,
            'save_video': True,
            'save_log': True
        },
        'training': {
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.01,
            'patience': 20,
            'weight_decay': 0.0005
        }
    }