#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Alpha Agent Network application.
This launches the GUI and initializes the application components.
"""

import sys
import os
import logging
from pathlib import Path

# Add the parent directory to the path to resolve imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Application imports
from agent.gui.app import AlphaAgentApp
from agent.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'agent.log'))
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application"""
    try:
        logger.info("Starting Alpha Agent Network application")
        
        # Ensure required directories exist
        Config.ensure_directories()
        
        # Initialize and start the GUI
        app = AlphaAgentApp()
        app.start()
        
    except Exception as e:
        logger.exception("Error starting the application")
        raise

if __name__ == "__main__":
    main() 