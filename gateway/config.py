import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

class Config:
    """Base configuration"""
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    GATEWAY_PORT = int(os.getenv('GATEWAY_PORT', 5000))
    
    # Microservice URLs
    OPTIONS_BUILDER_SERVICE_URL = os.getenv('OPTIONS_BUILDER_SERVICE_URL', 'http://localhost:5001')
    
    # External API Keys
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    POLYGON_KEY_ID = os.getenv('POLYGON_KEY_ID')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 