"""Flask application factory."""

from flask import Flask, jsonify
import os

from src.config import get_config
from src.utils import setup_logging, get_logger
from src.db import DatabaseManager
from src.api import register_error_handlers, api_v1
from src.web import web

logger = get_logger(__name__)


def create_app(config_env=None):
    """
    Create and configure Flask application
    
    Args:
        config_env: Environment configuration ('development', 'testing', 'production')
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__, template_folder='templates')
    
    # Load configuration
    if config_env is None:
        config_env = os.environ.get('FLASK_ENV', 'development')
    
    config = get_config(config_env)
    app.config.from_object(config)
    
    # Setup logging
    setup_logging(
        app.config['LOG_FILE'],
        app.config['LOG_LEVEL'],
        app.config['LOG_MAX_BYTES'],
        app.config['LOG_BACKUP_COUNT']
    )
    
    logger.info(f"Creating Flask app with config: {config_env}")
    
    # Initialize application
    config.init_app(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register blueprints
    app.register_blueprint(web)
    app.register_blueprint(api_v1)
    
    # Initialize database
    init_database(app)
    
    # Add root health check
    @app.route('/health', methods=['GET'])
    def root_health():
        return jsonify({'status': 'ok', 'version': '1.0.0'}), 200
    
    # Add info endpoint
    @app.route('/info', methods=['GET'])
    def app_info():
        return jsonify({
            'name': 'License Plate Detection System',
            'version': '1.0.0',
            'environment': config_env,
            'debug': app.config['DEBUG']
        }), 200
    
    logger.info(f"Flask app created successfully for {config_env}")
    
    return app


def init_database(app):
    """Initialize database"""
    try:
        db_manager = DatabaseManager(app.config['DATABASE_PATH'])
        if not db_manager.initialize():
            logger.error("Failed to initialize database")
        else:
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


if __name__ == '__main__':
    app = create_app()
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    app.run(host=host, port=port, debug=app.config['DEBUG'])
