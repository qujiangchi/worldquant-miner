#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration module for the Alpha Agent Network application.
Defines settings, paths, and configuration options.
"""

import os
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List

@dataclass
class Config:
    """Configuration class for the Alpha Agent Network application"""
    
    # GUI Configuration
    WINDOW_TITLE: str = "Alpha Agent Network"
    WINDOW_SIZE: str = "1280x800"
    
    # Agent Configuration
    MAX_AGENTS: int = 5
    AGENT_CONCURRENCY: int = 3
    REQUEST_TIMEOUT: int = 30
    
    # API Configuration
    OPENAI_API_KEY: str = ""
    MOONSHOT_API_KEY: str = ""
    ALPHAVANTAGE_API_KEY: str = ""
    
    # Crawler Configuration
    CRAWL_DELAY: float = 0.5  # Seconds between requests
    MAX_PAGES_PER_SITE: int = 100
    USER_AGENT: str = "AlphaAgentBot/1.0"
    
    # Model Configuration
    DEFAULT_MODEL: str = "gpt-4o"
    TEXT_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # File paths
    CONFIG_FILE: str = os.path.join(os.path.dirname(__file__), "config.json")
    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
    AGENTS_DIR: str = os.path.join(os.path.dirname(__file__), "data", "agents")
    CRAWL_DIR: str = os.path.join(os.path.dirname(__file__), "data", "crawl")
    ALPHA_DIR: str = os.path.join(os.path.dirname(__file__), "data", "alphas")
    LOGS_DIR: str = os.path.join(os.path.dirname(__file__), "logs")
    
    # Research Sources - Using default_factory to avoid mutable default
    DEFAULT_SOURCES: List[str] = field(default_factory=lambda: [
        "https://www.quantopian.com/posts",
        "https://www.quantconnect.com/tutorials",
        "https://www.alphaarchitect.com/blog",
        "https://www.factorresearch.com/research",
        "https://www.findingalpha.com"
    ])
    
    @classmethod
    def load_from_file(cls) -> 'Config':
        """Load config from file or return default config if file doesn't exist"""
        config_path = cls.CONFIG_FILE
        
        if not os.path.exists(config_path):
            default_config = cls()
            default_config.save_to_file()
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Filter out any keys that don't exist in the class
            valid_keys = {field for field in cls.__dataclass_fields__}
            filtered_data = {k: v for k, v in config_data.items() if k in valid_keys}
            
            return cls(**filtered_data)
            
        except Exception as e:
            print(f"Error loading config: {e}")
            return cls()
    
    def save_to_file(self) -> None:
        """Save configuration to file"""
        config_path = self.CONFIG_FILE
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    @staticmethod
    def ensure_directories() -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            Config.DATA_DIR,
            Config.AGENTS_DIR,
            Config.CRAWL_DIR,
            Config.ALPHA_DIR,
            Config.LOGS_DIR
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

# Initialize config
config = Config.load_from_file() 