# QURE Deployment Guide

Complete guide for deploying QURE in development and production environments.

## Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose** (for infrastructure)
- **Git**
- **8GB+ RAM** (recommended)
- **API Keys**: OpenAI or Anthropic (for GenAI agent)

---

## Quick Start (Development)

### 1. Clone Repository

```bash
git clone https://github.com/mpointer/QURE.git
cd QURE
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install poetry
poetry install
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your_key_here
# Or: ANTHROPIC_API_KEY=your_key_here
```

### 4. Start Infrastructure (Optional)

QURE can run without infrastructure for basic demos, but for full functionality:

```bash
cd docker
docker-compose up -d

# Verify services
docker-compose ps
```

Services:
- **Redis** (port 6379) - Message queue & cache
- **Neo4j** (port 7474, 7687) - Graph database
- **ChromaDB** (port 8000) - Vector store
- **PostgreSQL** (port 5432) - Feature store
- **Temporal** (port 7233, 8233) - Workflow orchestration
- **MLflow** (port 5000) - Model registry
- **Grafana** (port 3000) - Monitoring

### 5. Generate Test Data

```bash
python data/synthetic/generate_finance_data.py
```

### 6. Run Demo

```bash
# Command-line demo
python demos/finance_reconciliation_demo.py

# Or: Streamlit UI
cd ui
streamlit run streamlit_app.py
```

The UI will open at http://localhost:8501

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_algorithm_agent.py -v

# Run with coverage
pytest tests/ --cov=agents --cov-report=html
```

---

## Production Deployment

### Option 1: Docker Deployment

#### Build Images

```bash
# Build all images
docker-compose -f docker/docker-compose.prod.yml build

# Tag images
docker tag qure/api:latest qure/api:1.0.0
docker tag qure/ui:latest qure/ui:1.0.0
docker tag qure/workers:latest qure/workers:1.0.0
```

#### Deploy

```bash
# Deploy full stack
docker-compose -f docker/docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker/docker-compose.prod.yml logs -f api

# Scale workers
docker-compose -f docker/docker-compose.prod.yml up -d --scale workers=3
```

### Option 2: Kubernetes Deployment

#### Prerequisites

- Kubernetes cluster (GKE, EKS, AKS, or local minikube)
- kubectl configured
- Helm 3+

#### Deploy with Helm

```bash
# Add QURE helm repo
helm repo add qure https://charts.qure.ai
helm repo update

# Install QURE
helm install qure qure/qure \
  --namespace qure \
  --create-namespace \
  --set api.replicas=3 \
  --set workers.replicas=5 \
  --set ingress.enabled=true \
  --set ingress.host=qure.yourdomain.com

# Check status
kubectl get pods -n qure
```

#### Manual K8s Deployment

```bash
# Create namespace
kubectl create namespace qure

# Create secrets
kubectl create secret generic qure-secrets \
  --from-literal=openai-api-key=$OPENAI_API_KEY \
  --from-literal=db-password=$DB_PASSWORD \
  -n qure

# Apply manifests
kubectl apply -f k8s/ -n qure

# Check deployment
kubectl get all -n qure
```

### Option 3: Cloud Platforms

#### AWS Deployment

```bash
# Using AWS App Runner
aws apprunner create-service \
  --service-name qure-api \
  --source-configuration ImageRepository={...} \
  --instance-configuration Cpu=1024,Memory=2048

# Or: ECS Fargate
aws ecs create-service \
  --cluster qure-cluster \
  --service-name qure-api \
  --task-definition qure-api:1 \
  --desired-count 2
```

#### Azure Deployment

```bash
# Using Azure Container Apps
az containerapp create \
  --name qure-api \
  --resource-group qure-rg \
  --image qure/api:latest \
  --target-port 8000 \
  --ingress external
```

#### GCP Deployment

```bash
# Using Cloud Run
gcloud run deploy qure-api \
  --image gcr.io/your-project/qure-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi
```

---

## Configuration

### Environment Variables

Required:
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` - LLM provider
- `DATABASE_URL` - PostgreSQL connection string (production)
- `NEO4J_URI` - Neo4j connection URI (production)
- `CHROMA_HOST` - ChromaDB host (production)

Optional:
- `LOG_LEVEL` - Logging level (default: INFO)
- `MAX_WORKERS` - Worker pool size (default: 4)
- `REDIS_URL` - Redis connection string
- `TEMPORAL_HOST` - Temporal server host

### Configuration Files

- `pyproject.toml` - Python dependencies
- `docker-compose.yml` - Development infrastructure
- `docker-compose.prod.yml` - Production infrastructure
- `k8s/*.yaml` - Kubernetes manifests
- `.env.example` - Environment template

---

## Monitoring & Logging

### Application Logs

```bash
# View API logs
docker-compose logs -f api

# View worker logs
docker-compose logs -f workers

# View all logs
docker-compose logs -f
```

### Grafana Dashboards

Access Grafana at http://localhost:3000 (default credentials: admin/admin)

Dashboards:
- **QURE Overview** - System health and metrics
- **Agent Performance** - Individual agent statistics
- **Decision Analytics** - Policy decision trends
- **Infrastructure** - Database and queue metrics

### Custom Metrics

QURE exports Prometheus metrics at `/metrics`:
- `qure_cases_total` - Total cases processed
- `qure_agent_latency_seconds` - Agent execution time
- `qure_decision_distribution` - Decision type counts
- `qure_errors_total` - Error counts by type

---

## Troubleshooting

### Common Issues

#### 1. API Keys Not Working

```bash
# Verify environment
echo $OPENAI_API_KEY

# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### 2. Database Connection Failed

```bash
# Check Postgres container
docker-compose ps postgres

# Test connection
psql $DATABASE_URL -c "SELECT version();"

# View logs
docker-compose logs postgres
```

#### 3. ChromaDB Not Responding

```bash
# Restart ChromaDB
docker-compose restart chroma

# Check logs
docker-compose logs chroma

# Test connection
curl http://localhost:8000/api/v1/heartbeat
```

#### 4. Out of Memory

```bash
# Increase Docker memory limit in Docker Desktop settings

# Or: Reduce worker count
docker-compose up -d --scale workers=1
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m pdb demos/finance_reconciliation_demo.py
```

---

## Maintenance

### Database Backups

```bash
# Backup Postgres
docker-compose exec postgres pg_dump -U qure qure > backup.sql

# Backup Neo4j
docker-compose exec neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j-backup.dump

# Backup ChromaDB
docker cp qure_chroma_1:/chroma/chroma.sqlite3 ./chroma-backup.sqlite3
```

### Updates

```bash
# Pull latest code
git pull origin main

# Update dependencies
poetry update

# Rebuild images
docker-compose build

# Restart services
docker-compose up -d
```

### Scaling

```bash
# Scale API servers
docker-compose up -d --scale api=3

# Scale workers
docker-compose up -d --scale workers=5

# Auto-scaling with K8s
kubectl autoscale deployment qure-api \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n qure
```

---

## Security

### Best Practices

1. **API Keys**: Store in secrets manager (AWS Secrets Manager, Azure Key Vault, etc.)
2. **Database**: Use SSL/TLS connections in production
3. **Network**: Run behind VPC/private network
4. **Authentication**: Implement OAuth2 or JWT for API access
5. **Audit Logs**: Enable immutable audit logging
6. **Encryption**: Encrypt sensitive data at rest

### Securing Secrets

```bash
# AWS Secrets Manager
aws secretsmanager create-secret \
  --name qure/openai-key \
  --secret-string $OPENAI_API_KEY

# Kubernetes secrets
kubectl create secret generic qure-secrets \
  --from-literal=openai-api-key=$OPENAI_API_KEY \
  -n qure

# Docker secrets
echo $OPENAI_API_KEY | docker secret create openai_key -
```

---

## Performance Tuning

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_cases_status ON cases(status);
CREATE INDEX idx_cases_created_at ON cases(created_at DESC);

-- Vacuum and analyze
VACUUM ANALYZE cases;
```

### Caching

```python
# Enable Redis caching
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600  # 1 hour
```

### Worker Pool

```bash
# Adjust worker count based on CPU cores
MAX_WORKERS=8  # For 8-core machine
```

---

## Support

- **Issues**: https://github.com/mpointer/QURE/issues
- **Documentation**: https://docs.qure.ai
- **Email**: mpointer@gmail.com

---

## License

See LICENSE file for details.
