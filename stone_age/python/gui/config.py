import os
from dataclasses import dataclass
import json
from typing import Dict, Any

@dataclass
class Config:
    # GUI Configuration
    WINDOW_TITLE: str = "Alpha Mining Workbench"
    WINDOW_SIZE: str = "1200x800"
    
    # Alpha Mining Parameters
    MAX_ITERATIONS: int = 1000
    POPULATION_SIZE: int = 100
    MUTATION_RATE: float = 0.1
    CROSSOVER_RATE: float = 0.8
    
    # Data Parameters
    DATA_WINDOW: int = 252  # Trading days in a year
    MIN_SAMPLES: int = 1000
    
    # API Configuration
    API_URL: str = "https://api.example.com"
    API_KEY: str = ""
    
    # Authentication
    USERNAME: str = ""
    AUTH_TOKEN: str = ""
    
    # Default values
    DEFAULT_MAX_ITERATIONS: int = 1000
    DEFAULT_POPULATION_SIZE: int = 100
    DEFAULT_API_URL: str = "https://api.example.com"
    
    # File paths
    CONFIG_FILE: str = "config.json"
    RESULTS_DIR: str = "results"
    DATA_DIR: str = "data"
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        if not os.path.exists(filepath):
            return cls()
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def save_to_file(self, filepath: str) -> None:
        config_data = {
            'WINDOW_TITLE': self.WINDOW_TITLE,
            'WINDOW_SIZE': self.WINDOW_SIZE,
            'MAX_ITERATIONS': self.MAX_ITERATIONS,
            'POPULATION_SIZE': self.POPULATION_SIZE,
            'MUTATION_RATE': self.MUTATION_RATE,
            'CROSSOVER_RATE': self.CROSSOVER_RATE,
            'DATA_WINDOW': self.DATA_WINDOW,
            'MIN_SAMPLES': self.MIN_SAMPLES,
            'API_URL': self.API_URL,
            'API_KEY': self.API_KEY,
            'USERNAME': self.USERNAME,
            'AUTH_TOKEN': self.AUTH_TOKEN,
            'DEFAULT_MAX_ITERATIONS': self.DEFAULT_MAX_ITERATIONS,
            'DEFAULT_POPULATION_SIZE': self.DEFAULT_POPULATION_SIZE,
            'DEFAULT_API_URL': self.DEFAULT_API_URL,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=4)
    
    @staticmethod
    def ensure_directories():
        directories = [Config.RESULTS_DIR, Config.DATA_DIR]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory) 