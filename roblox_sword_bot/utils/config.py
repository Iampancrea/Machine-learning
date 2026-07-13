"""
Configuration loader for Roblox Sword Fight Bot
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Load and manage configuration settings"""
    
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Set device based on hardware settings
        if config['hardware']['use_gpu']:
            import torch
            config['hardware']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            config['hardware']['device'] = 'cpu'
            
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __repr__(self) -> str:
        return f"Config({self.config_path})"
    
    def save(self, path: str = None):
        """Save current configuration to file"""
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"Configuration saved to {save_path}")


def load_config(config_path: str = "configs/default_config.yaml") -> Config:
    """Convenience function to load configuration"""
    path = Path(config_path)
    if not path.is_absolute():
        project_root = Path(__file__).parent.parent
        # Try relative to cwd
        if not path.exists():
            # Try relative to project root
            test_path = project_root / config_path
            if test_path.exists():
                path = test_path
            # Fallback to configs/ directory
            elif not str(config_path).startswith("configs/") and not str(config_path).endswith(".yaml"):
                test_path = project_root / "configs" / f"{config_path}.yaml"
                if test_path.exists():
                    path = test_path
                    
    return Config(str(path))


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print(f"Device: {config['hardware']['device']}")
    print(f"Model type: {config['model']['type']}")
    print(f"Resolution: {config['capture']['resolution']}")
