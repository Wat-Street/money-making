from flask import Blueprint, request, jsonify, current_app
import requests
import os

# Create a Blueprint
options_strategy_bp = Blueprint('options_strategy_bp',
                                __name__,
                                url_prefix='/options-builder')

# Get service URL from environment variable with fallback
OPTIONS_BUILDER_API_URL = os.getenv('OPTIONS_BUILDER_SERVICE_URL', 'http://localhost:5001')

@options_strategy_bp.route('/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_options_builder(subpath):
    """
    Proxy all requests to the Options Strategy Builder microservice
    """
    url = f"{OPTIONS_BUILDER_API_URL}/{subpath}"

    try:
        # Forward the request
        resp = requests.request(
            method=request.method,
            url=url,
            headers={key: value for (key, value) in request.headers if key != 'Host'},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=30  # Add timeout for better error handling
        )

        # Create a response to send back to the client
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in resp.raw.headers.items()
                   if name.lower() not in excluded_headers]

        # Handle response based on content type
        if resp.headers.get('Content-Type', '').startswith('application/json'):
            try:
                flask_response = jsonify(resp.json())
                flask_response.status_code = resp.status_code
            except ValueError:
                # If JSON parsing fails, treat as plain text
                flask_response = current_app.response_class(
                    response=resp.content,
                    status=resp.status_code,
                    mimetype='text/plain'
                )
        else:
            flask_response = current_app.response_class(
                response=resp.content,
                status=resp.status_code,
                mimetype=resp.headers.get('Content-Type', 'text/plain')
            )
        
        # Add headers to the Flask response object
        for name, value in headers:
            flask_response.headers.set(name, value)

        return flask_response

    except requests.exceptions.ConnectionError:
        current_app.logger.error(f"Failed to connect to Options Strategy Builder at {url}")
        return jsonify({
            "error": "Options Strategy Builder service is unavailable",
            "service_url": OPTIONS_BUILDER_API_URL,
            "requested_path": subpath
        }), 502
    except requests.exceptions.Timeout:
        current_app.logger.error(f"Timeout connecting to Options Strategy Builder at {url}")
        return jsonify({
            "error": "Options Strategy Builder service timeout",
            "service_url": OPTIONS_BUILDER_API_URL
        }), 504
    except Exception as e:
        current_app.logger.error(f"Error proxying to options builder: {e}", exc_info=True)
        return jsonify({
            "error": "Internal gateway error",
            "details": str(e)
        }), 500

@options_strategy_bp.route('/health', methods=['GET'])
def options_strategy_health():
    """
    Health check for the Options Strategy Builder service
    """
    try:
        resp = requests.get(f"{OPTIONS_BUILDER_API_URL}/api/v1/ping", timeout=5)
        if resp.status_code == 200:
            return jsonify({
                "service": "options-strategy-builder",
                "status": "healthy",
                "upstream_response": resp.json()
            }), 200
        else:
            return jsonify({
                "service": "options-strategy-builder", 
                "status": "unhealthy",
                "upstream_status": resp.status_code
            }), 503
    except Exception as e:
        return jsonify({
            "service": "options-strategy-builder",
            "status": "unavailable",
            "error": str(e)
        }), 503 