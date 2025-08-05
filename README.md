# SAP S/4HANA PM Data Quality Workflow

A modular, containerized data quality workflow for SAP S/4HANA Plant Maintenance (PM) module that automatically assesses, reports, and improves data quality across PM datasets.

## 🏗️ Architecture

The system consists of the following Docker containers:

- **Mock SAP Service** (Port 8000): Simulates SAP S/4HANA PM data extraction for development
- **Validation Engine** (Port 8001): Runs data quality rules and validations
- **Logging & Reporting Service** (Port 8080): Web dashboard and REST API
- **Orchestrator** (Port 8081): Apache Airflow for workflow scheduling
- **PostgreSQL Database** (Port 5432): Stores validation results and metrics
- **pgAdmin** (Port 5050): Database administration interface
- **Redis** (Port 6379): Caching and message broker for Airflow
- **Prometheus** (Port 9090): Metrics collection and monitoring
- **Grafana** (Port 3000): Data visualization and dashboards

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- SAP S/4HANA system access (for production)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YavinOwens/Data-Quality---Plant-Maintenance-.git
   cd Data-Quality---Plant-Maintenance-
   ```

2. **Configure environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your SAP connection details
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Access the services**
   - **Dashboard**: http://localhost:8080
   - **Grafana**: http://localhost:3000 (admin/admin)
   - **pgAdmin**: http://localhost:5050 (admin@sap-data-quality.com/admin)
   - **Prometheus**: http://localhost:9090
   - **Airflow**: http://localhost:8081

## 📊 Data Quality Dimensions

The system validates data across four key dimensions:

### 1. Completeness
- Missing required fields (Equipment, EquipmentName, etc.)
- Empty strings and null values
- Required business data presence

### 2. Accuracy
- Valid equipment categories (M, F, K, P)
- Proper serial number formats
- Valid functional location hierarchies
- Correct measurement units

### 3. Consistency
- Cross-reference validation between datasets
- Equipment-functional location relationships
- Maintenance order-status consistency
- Hierarchical data integrity

### 4. Timeliness
- Overdue maintenance orders
- Incomplete order status tracking
- Notification response times
- Plan execution monitoring

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# SAP Connection
SAP_HOST=your-sap-server.com
SAP_CLIENT=100
SAP_USERNAME=your_sap_user
SAP_PASSWORD=your_sap_password

# Database
DB_NAME=sap_data_quality
DB_USER=sap_user
DB_PASSWORD=your_db_password

# pgAdmin
PGADMIN_EMAIL=admin@sap-data-quality.com
PGADMIN_PASSWORD=admin

# Security
SECRET_KEY=your-secret-key-here
```

### Data Quality Rules

Rules are configured in YAML format under `config/rules/`:

- `equipment_completeness.yaml`: Equipment master data validation
- `maintenance_orders_timeliness.yaml`: Maintenance order timeliness
- `cross_reference_consistency.yaml`: Cross-dataset consistency

## 📈 Monitoring & Administration

### pgAdmin Database Administration

Access pgAdmin at http://localhost:5050 to:

- **Browse database schema**: View tables, views, and relationships
- **Execute SQL queries**: Run custom queries for data analysis
- **Monitor performance**: Check query execution plans and statistics
- **Manage data**: Import/export data, backup/restore operations
- **User management**: Create/modify database users and permissions

**Default pgAdmin credentials**:
- Email: `admin@sap-data-quality.com`
- Password: `admin`

**Pre-configured database connection**:
- Host: `postgres` (Docker service name)
- Port: `5432`
- Database: `sap_data_quality`
- Username: `sap_user`

### Grafana Dashboards

Access Grafana at http://localhost:3000 for:

- **Data Quality Metrics**: Success rates, error counts, validation trends
- **System Performance**: Container health, resource usage, response times
- **Business KPIs**: Equipment availability, maintenance efficiency
- **Custom Dashboards**: Create custom visualizations

### Prometheus Metrics

Access Prometheus at http://localhost:9090 for:

- **System Metrics**: Container health, resource utilization
- **Application Metrics**: API response times, error rates
- **Business Metrics**: Data quality scores, validation results
- **Custom Queries**: PromQL queries for custom analysis

## 🔄 Workflow

1. **Data Extraction**: Mock SAP Service simulates PM data extraction
2. **Validation**: Validation Engine applies data quality rules
3. **Storage**: Results stored in PostgreSQL database
4. **Reporting**: Logging Service provides web dashboard and API
5. **Monitoring**: Prometheus/Grafana track system health and metrics
6. **Administration**: pgAdmin provides database management interface

## 🛠️ Development

### Local Development Setup

```bash
# Create virtual environment
python3 -m venv sap_data_quality_env
source sap_data_quality_env/bin/activate

# Install dependencies
pip install -r requirements-test.txt

# Run tests
python test_setup.py
python test_workflow.py
```

### Testing

```bash
# Test environment setup
python test_setup.py

# Test complete workflow
python test_workflow.py

# Test individual components
curl http://localhost:8080/health
curl http://localhost:8000/health
```

## 📋 API Endpoints

### Logging Service (Port 8080)

- `GET /health` - Health check
- `GET /api/validation-summary` - Validation summary
- `GET /api/recent-issues` - Recent data quality issues
- `GET /api/quality-report` - Comprehensive quality report
- `GET /api/export/csv` - Export data as CSV
- `GET /api/export/excel` - Export data as Excel
- `GET /api/export/pdf` - Export data as PDF

### Mock SAP Service (Port 8000)

- `GET /health` - Health check
- `GET /extract/equipment` - Extract equipment data
- `GET /extract/functional-locations` - Extract functional locations
- `GET /extract/maintenance-orders` - Extract maintenance orders
- `GET /extract/all` - Extract all PM data

### Validation Engine (Port 8001)

- `GET /health` - Health check
- `POST /validate/equipment` - Validate equipment data
- `POST /validate/maintenance-orders` - Validate maintenance orders
- `GET /rules` - List available validation rules

## 🔒 Security

- **Environment Variables**: Sensitive data stored in `.env` file
- **Network Isolation**: Services communicate via Docker network
- **SSL/TLS**: Encrypted communication with SAP (configurable)
- **Authentication**: OAuth 2.0 or certificate-based auth for SAP
- **Database Security**: PostgreSQL with user authentication

## 📚 Documentation

- [SAP Integration Guide](SAP_INTEGRATION_GUIDE.md) - Configure for real SAP systems
- [Architecture Documentation](ARCHITECTURE.md) - Detailed technical architecture
- [Data Quality Rules](config/rules/) - Rule configuration examples
- [Audit Documentation](audit.md) - SOC2 and ISO27001 compliance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the [troubleshooting section](SAP_INTEGRATION_GUIDE.md#troubleshooting)
2. Review the [architecture documentation](ARCHITECTURE.md)
3. Open an issue in the repository

## 🔄 Roadmap

- [ ] Kubernetes deployment manifests
- [ ] Helm charts for easy deployment
- [ ] Additional data quality rules
- [ ] Machine learning-based anomaly detection
- [ ] Real-time streaming validation
- [ ] Advanced reporting features
