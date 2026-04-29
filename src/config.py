"""
Configuration management for the License Plate Detection System
Supports multiple environments: development, testing, production
"""

import os
from datetime import timedelta
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    TESTING = False
    
    # Upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', str(BASE_DIR / 'uploads'))
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 200 * 1024 * 1024))  # 200MB (supports video)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv', 'webm'}
    VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    # Reduced from 60 → 30 for much faster video processing
    VIDEO_MAX_FRAMES = int(os.environ.get('VIDEO_MAX_FRAMES', 30))
    # Downscale video frames to this width before inference (matches YOLO fast_mode imgsz=640)
    VIDEO_FRAME_MAX_WIDTH = int(os.environ.get('VIDEO_FRAME_MAX_WIDTH', 640))
    
    # Database settings
    DATABASE_PATH = os.environ.get('DATABASE_PATH', str(BASE_DIR / 'data' / 'app.db'))
    DATABASE_URL = os.environ.get('DATABASE_URL', f'sqlite:///{DATABASE_PATH}')
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH', str(BASE_DIR / 'weights' / 'best.pt'))
    FALLBACK_MODEL = 'yolov8n.pt'
    DEFAULT_CONFIDENCE = float(os.environ.get('DEFAULT_CONFIDENCE', 0.20))
    MODEL_DEVICE = os.environ.get('MODEL_DEVICE', 'cpu')  # 'cpu' or 'cuda'
    
    # OCR settings
    OCR_LANGUAGE = os.environ.get('OCR_LANGUAGE', 'en')
    OCR_USE_GPU = os.environ.get('OCR_USE_GPU', 'False').lower() == 'true'
    OCR_ALLOWED_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=int(os.environ.get('SESSION_TIMEOUT', 1)))
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_DIR = os.environ.get('LOG_DIR', str(BASE_DIR / 'logs'))
    LOG_FILE = os.path.join(LOG_DIR, 'app.log')
    LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 10 * 1024 * 1024))  # 10MB
    LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 5))
    
    # API settings
    API_VERSION = 'v1'
    API_TITLE = 'License Plate Detection API'
    API_DESCRIPTION = 'Detection and recognition of license plates using YOLOv10 and EasyOCR'
    
    # Performance settings
    CACHE_TIMEOUT = int(os.environ.get('CACHE_TIMEOUT', 300))  # 5 minutes
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 4))
    REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', 60))  # seconds
    
    # Security settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5000')
    RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'True').lower() == 'true'
    RATE_LIMIT_REQUESTS = int(os.environ.get('RATE_LIMIT_REQUESTS', 100))
    RATE_LIMIT_PERIOD = int(os.environ.get('RATE_LIMIT_PERIOD', 3600))  # seconds
    
    # Files to ignore in uploads
    FORBIDDEN_FILES = {'.exe', '.bat', '.sh', '.cmd', '.com', '.pif', '.scr'}
    
    @classmethod
    def allowed_file(cls, filename):
        """Check if file is allowed"""
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in cls.ALLOWED_EXTENSIONS and ext not in cls.FORBIDDEN_FILES
    
    @classmethod
    def init_app(cls, app):
        """Initialize application with config"""
        # Ensure required directories exist
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.DATABASE_PATH), exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SESSION_COOKIE_SECURE = True
    
    @classmethod
    def init_app(cls, app):
        """Initialize production app"""
        Config.init_app(app)
        
        # Ensure security settings are strict
        if not cls.SECRET_KEY:
            raise ValueError('SECRET_KEY must be set in production')
        
        # Log production startup
        import logging
        logging.getLogger(__name__).info('Production environment initialized')


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get configuration for the specified environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
