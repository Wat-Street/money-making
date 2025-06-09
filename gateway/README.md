# WatStreet API Gateway

This is the central API Gateway for all WatStreet financial microservices. It acts as a single entry point for clients and proxies requests to appropriate microservices.

## Architecture

```
Client → API Gateway → Microservice A
                   → Microservice B
                   → Microservice C
```

## Setup

1. **Install Dependencies**
   ```bash
   cd gateway
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   Copy the environment variables from your existing `.env.local` file or create a new `.env` file:
   ```bash
   # API Gateway Configuration
   DEBUG=True
   GATEWAY_PORT=5000

   # Microservice URLs
   OPTIONS_BUILDER_SERVICE_URL=http://localhost:5001

   # External API Keys (if needed by services)
   POLYGON_API_KEY=your_polygon_api_key_here
   POLYGON_KEY_ID=your_polygon_key_id_here
   ```

3. **Start the Gateway**
   ```bash
   python app.py
   ```

## Available Services

### Options Strategy Builder
- **Prefix**: `/options-builder/*`
- **Port**: 5001
- **Health Check**: `GET /options-builder/health`

## Gateway Endpoints

- **Health Check**: `GET /health`
- **Gateway Info**: `GET /api/info`

## Adding New Services

1. Create a new blueprint in `blueprints/`
2. Import and register it in `app.py`
3. Add the service URL to configuration
4. Update this README

## Example Requests

```bash
# Gateway health check
curl http://localhost:5000/health

# Service health checks
curl http://localhost:5000/options-builder/health

# Options strategy builder (example)
curl http://localhost:5000/options-builder/api/v1/ping
``` 