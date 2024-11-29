from typing import Dict, Any
import yaml
from pathlib import Path
import logging
from dataclasses import dataclass

@dataclass
class ConfigManager:
    """Manages loading and accessing configuration settings."""
    config_path: str
    config: Dict[str, Any] = None

    def __post_init__(self):
        self.load_config()

    def load_config(self):
        """Loads the YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            self.config = {}

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