"""
Error handling for the API
"""

from flask import jsonify
import logging

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base API error"""
    
    def __init__(self, message: str, status_code: int = 400, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload
    
    def to_dict(self):
        """Convert to dictionary"""
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['success'] = False
        return rv


class ValidationError(APIError):
    """Validation error"""
    def __init__(self, message: str):
        super().__init__(message, 422)


class NotFoundError(APIError):
    """Resource not found error"""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, 404)


class ServerError(APIError):
    """Server error"""
    def __init__(self, message: str = "Internal server error"):
        super().__init__(message, 500)


def register_error_handlers(app):
    """Register error handlers with Flask app"""
    
    @app.errorhandler(APIError)
    def handle_api_error(error):
        """Handle API errors"""
        logger.warning(f"API Error: {error.message}")
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response
    
    @app.errorhandler(400)
    def handle_bad_request(error):
        """Handle 400 Bad Request"""
        logger.warning(f"Bad Request: {str(error)}")
        return jsonify({
            'message': 'Bad Request',
            'success': False
        }), 400
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 Not Found"""
        return jsonify({
            'message': 'Not Found',
            'success': False
        }), 404
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Handle 405 Method Not Allowed"""
        logger.warning(f"Method Not Allowed: {str(error)}")
        return jsonify({
            'message': 'Method Not Allowed',
            'success': False
        }), 405
    
    @app.errorhandler(500)
    def handle_server_error(error):
        """Handle 500 Internal Server Error"""
        logger.error(f"Server Error: {str(error)}")
        return jsonify({
            'message': 'Internal Server Error',
            'success': False
        }), 500
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Handle unexpected errors"""
        logger.error(f"Unexpected Error: {str(error)}", exc_info=True)
        return jsonify({
            'message': 'Internal Server Error',
            'success': False
        }), 500
