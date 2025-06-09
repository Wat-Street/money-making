# WatStreet Money Making - API Gateway

A financial analysis platform built with an API Gateway that proxies requests to microservices.

## Architecture Overview

```
Client Applications
        ↓
   API Gateway (Port 5000) __________
        ↓                           ↓
Options Builder Service       Other Services...
    (Port 5001)                  (Port XXXX)
```

## Repository Structure

```
money-making/
├── gateway/                     # API Gateway
│   ├── app.py                  # Main gateway application
│   ├── blueprints/             # Service proxy blueprints
│   │   ├── options_strategy_api.py
│   │   └── mean_reversion_api.py
│   ├── config.py               # Configuration management
│   ├── requirements.txt        # Gateway dependencies
│   └── README.md              # Gateway documentation
│
├── microservices/              # Future microservices directory
│
├── projects/                   # Legacy projects (to be migrated)
│   ├── options-volatility-model/
│   ├── crypto-arb-cross-exchange/
│   └── ...
│
└── server/                     # Legacy server code (deprecated)
```

## Quick Start

### 1. Start the API Gateway

```bash
cd gateway
pip install -r requirements.txt
# Copy your existing .env.local values to a new .env file
python app.py
```

The gateway will start on http://localhost:5000

### 2. Start Options Strategy Service (External)

This service should be running separately on port 5001. If you don't have it set up yet, you can create a simple one:

```python
# In a separate repository or directory
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/v1/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Options Strategy Builder API is alive!'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
```

## API Usage

### Via API Gateway (Recommended)

```bash
# Health checks
curl http://localhost:5000/health
curl http://localhost:5000/options-builder/health

# Options strategy builder (example)
curl http://localhost:5000/options-builder/api/v1/ping
```

### Direct Service Access (Development)

```bash
# Direct to options builder service
curl http://localhost:5001/api/v1/ping
```

## Migration from Legacy Code

The existing `server/` directory contains the original monolithic application. Key components have been extracted:

- **Options Strategy Proxy**: Moved to `gateway/blueprints/options_strategy_api.py`
- **API Gateway**: New implementation in `gateway/app.py`

### What Was Migrated

1. **Options Strategy Proxy**: 
   - Gateway now proxies all `/options-builder/*` requests to the external Options Strategy Builder service

## Adding New Services

To add a new microservice:

1. **Create the microservice**:
   ```bash
   mkdir microservices/your-service
   cd microservices/your-service
   # Create app.py with Flask application
   # Add /api/v1/ping health check endpoint
   ```

2. **Create the gateway blueprint**:
   ```bash
   # Create gateway/blueprints/your_service_api.py
   # Follow the pattern from existing blueprints
   ```

3. **Register the blueprint**:
   ```python
   # In gateway/app.py
   from blueprints.your_service_api import your_service_bp
   app.register_blueprint(your_service_bp)
   ```

4. **Update configuration**:
   ```bash
   # Add service URL to gateway/.env
   YOUR_SERVICE_URL=http://localhost:5003
   ```

## Environment Variables

### Gateway `.env`
```bash
DEBUG=True
GATEWAY_PORT=5000
OPTIONS_BUILDER_SERVICE_URL=http://localhost:5001
POLYGON_API_KEY=your_key_here
```

## Development Workflow

1. **Local Development**: Run all services locally on different ports
2. **Integration Testing**: Use the gateway to test service interactions
3. **Service Independence**: Each service can be developed and deployed independently
4. **Scalability**: Services can be scaled individually based on demand

## Legacy Support

The original `server/app.py` is preserved for reference but should not be used in production. All new development should follow the microservices pattern.

## Next Steps

1. Migrate remaining projects from `projects/` to microservices
2. Add authentication/authorization to the gateway
3. Implement proper logging and monitoring
4. Add service discovery and load balancing
5. Containerize services with Docker