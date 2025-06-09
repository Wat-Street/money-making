from flask import Flask, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

from blueprints.options_strategy_api import options_strategy_bp
# Add more blueprints as you add more microservices

# Load environment variables
load_dotenv('.env')

def create_app():
    app = Flask(__name__)
    
    # Enable CORS for all domains on all routes
    CORS(app)
    
    # Configuration
    app.config['DEBUG'] = os.getenv('DEBUG', 'True').lower() == 'true'
    
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
    app = create_app()
    port = int(os.getenv('GATEWAY_PORT', 5000))
    
    try:
        print(f"Starting WatStreet API Gateway on port {port}...")
        print("Available services:")
        print("  - Options Strategy Builder: /options-builder/*")
        print(f"  - Health Check: http://localhost:{port}/health")
        print(f"  - Gateway Info: http://localhost:{port}/api/info")
        
        app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])
        
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Make sure all required packages are installed: pip install -r requirements.txt")
    except Exception as e:
        import traceback
        print(f"Error starting gateway: {e}")
        traceback.print_exc() 