from typing import Dict, Any
import yaml
from pathlib import Path
import logging
from dataclasses import dataclass

@dataclass
class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or str(Path(__file__).parent / 'config.yaml')
        self.config: Dict[str, Any] = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            raise
            
    @property
    def okx_config(self) -> Dict[str, Any]:
        return self.config.get('okx', {})
        
    @property
    def data_collection_config(self) -> Dict[str, Any]:
        return self.config.get('data_collection', {})
        
    @property
    def signal_generation_config(self) -> Dict[str, Any]:
        return self.config.get('signal_generation', {})
        
    @property
    def funds_management_config(self) -> Dict[str, Any]:
        return self.config.get('funds_management', {})
        
    @property
    def monitoring_config(self) -> Dict[str, Any]:
        return self.config.get('monitoring', {}) 