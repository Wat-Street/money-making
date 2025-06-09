# WatStreet API Gateway Setup Guide

This guide will help you set up the API Gateway for the Options Strategy Builder service.

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Your existing API keys (Polygon, etc.)

## Quick Setup

### Option 1: Using Scripts (Recommended)

**For Windows:**
```cmd
# Start Gateway
scripts\start-gateway.bat
```

**For Linux/Mac:**
```bash
# Make scripts executable (if needed)
chmod +x scripts/*.sh

# Start Gateway  
./scripts/start-gateway.sh
```

### Option 2: Manual Setup

1. **Setup API Gateway**
   ```bash
   cd gateway
   python -m venv venv
   
   # Windows  
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   
   pip install -r requirements.txt
   python app.py
   ```

## Environment Configuration

1. **Copy your existing API keys**
   ```bash
   # Copy your .env.local or .env file to gateway/.env
   cp .env.local gateway/.env
   ```

2. **Verify environment variables**
   Make sure your `gateway/.env` file contains:
   ```
   DEBUG=True
   GATEWAY_PORT=5000
   OPTIONS_BUILDER_SERVICE_URL=http://localhost:5001
   POLYGON_API_KEY=your_actual_key_here
   ```

## Verification

Once the gateway is running, test the setup:

```bash
# Test Gateway
curl http://localhost:5000/health

# Test Options Strategy Builder via Gateway (requires external service on port 5001)
curl http://localhost:5000/options-builder/health
curl http://localhost:5000/options-builder/api/v1/ping
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Kill processes on port 5000
   # Windows
   netstat -ano | findstr :5000
   taskkill /PID <PID> /F
   
   # Linux/Mac
   lsof -ti:5000 | xargs kill -9
   ```

2. **Missing dependencies**
   ```bash
   # Install gateway dependencies
   pip install flask flask-cors requests python-dotenv
   ```

4. **Environment variables not loaded**
   - Make sure `.env` file is in the `gateway/` directory
   - Verify the file has no syntax errors
   - Check file permissions

## Migration Status

✅ **Completed:**
- API Gateway setup
- Options Strategy blueprint (proxy ready)
- Documentation and setup scripts

📋 **Future Enhancements:**
- Add authentication/authorization
- Add logging and monitoring
- Containerize with Docker

## Development Workflow

1. **Adding new microservices**: Create new services and corresponding gateway blueprints
2. **Modifying gateway routes**: Edit `gateway/blueprints/`
3. **Configuration changes**: Update `gateway/config.py` and `.env` files

## Legacy Code

The original `server/app.py` is preserved but should not be used. All new development should use the microservices architecture.