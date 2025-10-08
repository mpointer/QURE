#!/bin/bash

# QURE Environment Setup Script

echo "========================================"
echo "QURE Environment Setup"
echo "========================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher required. Found: $python_version"
    exit 1
fi
echo "✅ Python version: $python_version"

# Check Docker
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop."
    exit 1
fi
echo "✅ Docker found: $(docker --version)"

# Check Docker Compose
echo "Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi
echo "✅ Docker Compose found: $(docker-compose --version)"

# Create virtual environment
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Install Poetry
echo "Installing Poetry..."
pip install --upgrade pip
pip install poetry
echo "✅ Poetry installed"

# Install dependencies
echo "Installing Python dependencies..."
poetry install
echo "✅ Dependencies installed"

# Download spaCy model
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm
echo "✅ spaCy model downloaded"

# Create .env file
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created - PLEASE EDIT WITH YOUR API KEYS"
else
    echo "⚠️  .env file already exists"
fi

# Start Docker services
echo "Starting Docker services..."
cd docker
docker-compose up -d
cd ..
echo "✅ Docker services started"

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 10

# Verify services
echo "Verifying services..."
services=("redis:6379" "neo4j:7474" "chroma:8000" "postgres:5432")
for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if nc -z localhost "$port" 2>/dev/null; then
        echo "✅ $name is running on port $port"
    else
        echo "❌ $name is NOT running on port $port"
    fi
done

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys (OPENAI_API_KEY or ANTHROPIC_API_KEY)"
echo "2. Run a demo: python scripts/run_demo.py --vertical finance"
echo "3. View Temporal UI: http://localhost:8080"
echo "4. View MLflow UI: http://localhost:5000"
echo "5. View Grafana: http://localhost:3000 (admin/admin)"
echo "6. View Neo4j Browser: http://localhost:7474 (neo4j/password)"
echo ""
