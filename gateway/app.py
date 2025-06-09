from flask import Flask, jsonify
from flask_cors import CORS
import os

from blueprints.options_strategy_api import options_strategy_bp
from config import config
# Add more blueprints as you add more microservices

def create_app(config_name=None):
    app = Flask(__name__)
    
    # Load configuration
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    app.config.from_object(config[config_name])
    
    # Enable CORS for all domains on all routes
    CORS(app)
    
    # Register service blueprints
    app.register_blueprint(options_strategy_bp)
    
    # Gateway health check
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'API Gateway',
            'registered_services': [
                'options-builder'
            ]
        }), 200
    
    # Gateway info endpoint
    @app.route('/api/info', methods=['GET'])
    def gateway_info():
        return jsonify({
            'name': 'WatStreet Money Making API Gateway',
            'version': '1.0.0',
            'description': 'Central API Gateway for all WatStreet financial services',
            'available_services': {
                '/options-builder/*': 'Options Strategy Builder Service'
            }
        }), 200
    
    return app

if __name__ == '__main__':
    # Determine environment
    env = os.getenv('FLASK_ENV', 'default')
    app = create_app(env)
    
    try:
        print(f"Starting WatStreet API Gateway (env: {env})...")
        print(f"Debug mode: {app.config['DEBUG']}")
        print("Available services:")
        print("  - Options Strategy Builder: /options-builder/*")
        print(f"  - Health Check: http://localhost:{app.config['GATEWAY_PORT']}/health")
        print(f"  - Gateway Info: http://localhost:{app.config['GATEWAY_PORT']}/api/info")
        
        app.run(
            host='0.0.0.0', 
            port=app.config['GATEWAY_PORT'], 
            debug=app.config['DEBUG']
        )
        
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Make sure all required packages are installed: pip install -r requirements.txt")
    except Exception as e:
        import traceback
        print(f"Error starting gateway: {e}")
        traceback.print_exc() 