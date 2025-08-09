# 🚀 Quick Access Guide - SAP Data Quality Workflow

## 🌐 Service URLs

### Core Services
| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **Dashboard** | http://localhost:8080 | None | Main web dashboard and API |
| **pgAdmin** | http://localhost:5050 | admin@sap-data-quality.com / admin | Database administration |
| **Grafana** | http://localhost:3000 | admin / admin | Metrics visualization |
| **Prometheus** | http://localhost:9090 | None | Metrics collection |
| **Airflow** | http://localhost:8081 | None | Workflow orchestration |

### API Endpoints
| Service | URL | Purpose |
|---------|-----|---------|
| **SAP Connector** | http://localhost:8000 | Data extraction from SAP |
| **Validation Engine** | http://localhost:8001 | Data quality validation |

## 🔧 Current Status

### ✅ Running Services (6/8)
- **Logging Service** (Port 8080) - Web dashboard and API
- **PostgreSQL Database** (Port 5432) - Data storage
- **pgAdmin** (Port 5050) - Database administration
- **Prometheus** (Port 9090) - Metrics collection
- **Grafana** (Port 3000) - Data visualization
- **Redis** (Port 6379) - Caching and message broker

### ⚠️ Services Not Running (2/8)
- **SAP Connector** (Port 8000) - Needs to be started
- **Validation Engine** (Port 8001) - Needs to be started

## 🛠️ pgAdmin Database Administration

### Access pgAdmin
1. Open http://localhost:5050
2. Login with:
   - **Email**: admin@sap-data-quality.com
   - **Password**: admin

### Pre-configured Database
- **Server Name**: SAP Data Quality Database
- **Host**: postgres (automatically configured)
- **Port**: 5432
- **Database**: sap_data_quality
- **Username**: sap_user

### Common pgAdmin Tasks

#### 1. Browse Database Schema
- Expand "Servers" → "SAP Data Quality Database"
- View tables, views, and relationships

#### 2. Execute SQL Queries
- Right-click database → "Query Tool"
- Write and execute custom SQL queries

#### 3. View Data Quality Results
```sql
-- View recent validation results
SELECT * FROM validation_results 
ORDER BY created_at DESC 
LIMIT 10;

-- Check data quality metrics
SELECT 
    validation_type,
    COUNT(*) as total_validations,
    AVG(success_rate) as avg_success_rate
FROM validation_results 
GROUP BY validation_type;
```

#### 4. Export Data
- Right-click query results → "Download"
- Choose format: CSV, JSON, XML, etc.

## 📊 Dashboard Features

### Main Dashboard (http://localhost:8080)
- **Validation Summary**: Overview of data quality metrics
- **Recent Issues**: Latest data quality problems
- **Quality Reports**: Detailed analysis reports
- **Export Options**: Download data in various formats

### Grafana Dashboards (http://localhost:3000)
- **System Metrics**: Container health and performance
- **Data Quality Metrics**: Validation success rates
- **Business KPIs**: Equipment availability and maintenance efficiency

## 🔍 Testing the System

### Run Comprehensive Test
```bash
# Activate virtual environment
source sap_data_quality_env/bin/activate

# Run full test suite
python test_workflow.py
```

### Individual Service Tests
```bash
# Test dashboard
curl http://localhost:8080/health

# Test pgAdmin
curl http://localhost:5050/misc/ping

# Test database
docker exec sap-postgres pg_isready -U sap_user -d sap_data_quality
```

## 🚀 Quick Commands

### Start All Services
```bash
docker-compose up -d
```

### Start Specific Services
```bash
# Start pgAdmin only
docker-compose up -d pgadmin

# Start database only
docker-compose up -d postgres

# Start monitoring only
docker-compose up -d prometheus grafana
```

### View Service Status
```bash
docker-compose ps
```

### View Service Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs pgadmin
docker-compose logs postgres
```

## 🔧 Configuration

### Environment Variables
Copy and edit the environment file:
```bash
cp env.example .env
# Edit .env with your settings
```

### Key Configuration Options
```bash
# Database settings
DB_NAME=sap_data_quality
DB_USER=sap_user
DB_PASSWORD=your_db_password

# pgAdmin settings
PGADMIN_EMAIL=admin@sap-data-quality.com
PGADMIN_PASSWORD=admin

# SAP connection (for production)
SAP_HOST=your-sap-server.com
SAP_USERNAME=your_sap_user
SAP_PASSWORD=your_sap_password
```

## 📚 Documentation

- **[README.md](README.md)** - Complete project documentation
- **[SAP_INTEGRATION_GUIDE.md](SAP_INTEGRATION_GUIDE.md)** - Real SAP integration setup
- **[PGADMIN_INTEGRATION_SUMMARY.md](PGADMIN_INTEGRATION_SUMMARY.md)** - pgAdmin integration details

## 🆘 Troubleshooting

### Common Issues

#### pgAdmin Not Accessible
```bash
# Check if container is running
docker-compose ps pgadmin

# Restart pgAdmin
docker-compose restart pgadmin

# Check logs
docker-compose logs pgadmin
```

#### Database Connection Issues
```bash
# Test database connectivity
docker exec sap-postgres pg_isready -U sap_user -d sap_data_quality

# Restart database
docker-compose restart postgres
```

#### Dashboard Not Loading
```bash
# Check if service is running
curl http://localhost:8080/health

# Restart logging service
docker-compose restart logging-service
```

### Reset Everything
```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: This deletes all data)
docker-compose down -v

# Start fresh
docker-compose up -d
```

## 🎯 Next Steps

1. **Start Missing Services**: Launch SAP Connector and Validation Engine
2. **Configure SAP Connection**: Set up real SAP system integration
3. **Customize Dashboards**: Create business-specific visualizations
4. **Set Up Monitoring**: Configure alerts and notifications
5. **Production Deployment**: Implement security and backup procedures

---

**Need Help?** Check the troubleshooting section or review the detailed documentation files. 