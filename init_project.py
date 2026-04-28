"""
Initialize the project
Sets up required directories and runs initial setup
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.db import DatabaseManager
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def initialize_project():
    """Initialize project directories and database"""
    
    # Setup logging
    setup_logging(Config.LOG_FILE, Config.LOG_LEVEL)
    
    logger.info("Initializing project...")
    
    # Create required directories
    directories = [
        Config.UPLOAD_FOLDER,
        Config.LOG_DIR,
        os.path.dirname(Config.DATABASE_PATH)
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Initialize database
    logger.info("Initializing database...")
    db = DatabaseManager(Config.DATABASE_PATH)
    if db.initialize():
        logger.info("Database initialized successfully")
    else:
        logger.error("Failed to initialize database")
        return False
    
    # Check model file
    if not os.path.exists(Config.MODEL_PATH):
        logger.warning(f"Model file not found: {Config.MODEL_PATH}")
        logger.info(f"Please download the model and place it at: {Config.MODEL_PATH}")
    else:
        logger.info(f"Model file found: {Config.MODEL_PATH}")
    
    logger.info("Project initialization complete!")
    return True


if __name__ == '__main__':
    success = initialize_project()
    sys.exit(0 if success else 1)
