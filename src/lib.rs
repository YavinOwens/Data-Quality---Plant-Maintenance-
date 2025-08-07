use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use tokio::sync::mpsc;

// Core data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub sql_condition: String,
    pub severity: RuleSeverity,
    pub category: RuleCategory,
    pub created_by: String,
    pub created_at: DateTime<Utc>,
    pub is_active: bool,
    pub performance_metrics: RulePerformance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleCategory {
    DataQuality,
    BusinessLogic,
    Compliance,
    Performance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulePerformance {
    pub success_rate: f64,
    pub false_positive_rate: f64,
    pub execution_time_ms: u64,
    pub last_executed: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPattern {
    pub id: String,
    pub pattern_type: PatternType,
    pub confidence: f64,
    pub field_name: String,
    pub pattern_description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    Anomaly,
    Trend,
    Correlation,
    Completeness,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTest {
    pub id: String,
    pub name: String,
    pub rule_a: ValidationRule,
    pub rule_b: ValidationRule,
    pub start_date: DateTime<Utc>,
    pub end_date: Option<DateTime<Utc>>,
    pub status: TestStatus,
    pub metrics: TestMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestStatus {
    Running,
    Completed,
    Failed,
    Paused,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetrics {
    pub rule_a_performance: f64,
    pub rule_b_performance: f64,
    pub statistical_significance: f64,
    pub sample_size: u64,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingDataItem {
    pub id: String,
    pub entity_type: EntityType,
    pub entity_id: String,
    pub missing_fields: Vec<String>,
    pub priority: Priority,
    pub estimated_effort: u64, // in minutes
    pub business_impact: ImpactLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Equipment,
    FunctionalLocation,
    MaintenanceOrder,
    SparePart,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Minimal,
    Moderate,
    Significant,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcelScript {
    pub id: String,
    pub name: String,
    pub description: String,
    pub template_path: String,
    pub validation_rules: Vec<ValidationRule>,
    pub data_mapping: DataMapping,
    pub status: ScriptStatus,
    pub created_at: DateTime<Utc>,
    pub approved_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMapping {
    pub source_fields: Vec<String>,
    pub target_fields: Vec<String>,
    pub transformation_rules: Vec<TransformationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationRule {
    pub source_field: String,
    pub target_field: String,
    pub transformation_type: TransformationType,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationType {
    Direct,
    Format,
    Validate,
    Calculate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScriptStatus {
    Draft,
    PendingApproval,
    Approved,
    Rejected,
    Executed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub id: String,
    pub name: String,
    pub steps: Vec<WorkflowStep>,
    pub status: WorkflowStatus,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub id: String,
    pub name: String,
    pub agent_type: AgentType,
    pub parameters: HashMap<String, String>,
    pub status: StepStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    ValidationRule,
    ABTesting,
    MissingData,
    ScriptGeneration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkflowStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

// Agent trait
pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    async fn execute(&self, input: AgentInput) -> Result<AgentOutput, Box<dyn Error + Send + Sync>>;
    async fn get_status(&self) -> AgentStatus;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInput {
    pub data: HashMap<String, serde_json::Value>,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    pub success: bool,
    pub data: HashMap<String, serde_json::Value>,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub is_active: bool,
    pub last_execution: Option<DateTime<Utc>>,
    pub metrics: HashMap<String, f64>,
    pub error_count: u64,
}

// Validation Rule Agent
pub struct ValidationRuleAgent {
    model: Box<dyn RuleGenerationModel>,
    database: DatabaseConnection,
    rule_repository: ValidationRuleRepository,
}

impl ValidationRuleAgent {
    pub fn new(
        model: Box<dyn RuleGenerationModel>,
        database: DatabaseConnection,
        rule_repository: ValidationRuleRepository,
    ) -> Self {
        Self {
            model,
            database,
            rule_repository,
        }
    }

    pub async fn analyze_data_patterns(&self, dataset: &str) -> Result<Vec<DataPattern>, Box<dyn Error + Send + Sync>> {
        // Analyze data patterns using ML models
        let patterns = self.model.analyze_patterns(dataset).await?;
        Ok(patterns)
    }

    pub async fn generate_rules(&self, patterns: Vec<DataPattern>) -> Result<Vec<ValidationRule>, Box<dyn Error + Send + Sync>> {
        let mut rules = Vec::new();
        
        for pattern in patterns {
            let rule = ValidationRule {
                id: format!("rule_{}", uuid::Uuid::new_v4()),
                name: format!("Auto-generated rule for {}", pattern.field_name),
                description: pattern.pattern_description.clone(),
                sql_condition: self.generate_sql_condition(&pattern),
                severity: self.determine_severity(&pattern),
                category: RuleCategory::DataQuality,
                created_by: "AI Agent".to_string(),
                created_at: Utc::now(),
                is_active: true,
                performance_metrics: RulePerformance {
                    success_rate: 0.0,
                    false_positive_rate: 0.0,
                    execution_time_ms: 0,
                    last_executed: Utc::now(),
                },
            };
            rules.push(rule);
        }
        
        Ok(rules)
    }

    pub async fn optimize_rules(&self, rules: Vec<ValidationRule>) -> Result<Vec<ValidationRule>, Box<dyn Error + Send + Sync>> {
        let mut optimized_rules = Vec::new();
        
        for mut rule in rules {
            // Apply optimization logic based on performance metrics
            if rule.performance_metrics.success_rate < 0.8 {
                rule.sql_condition = self.optimize_condition(&rule.sql_condition);
            }
            optimized_rules.push(rule);
        }
        
        Ok(optimized_rules)
    }

    fn generate_sql_condition(&self, pattern: &DataPattern) -> String {
        match pattern.pattern_type {
            PatternType::Anomaly => format!("{} IS NOT NULL AND {} != ''", pattern.field_name, pattern.field_name),
            PatternType::Completeness => format!("{} IS NOT NULL", pattern.field_name),
            PatternType::Trend => format!("{} IS NOT NULL", pattern.field_name),
            PatternType::Correlation => format!("{} IS NOT NULL", pattern.field_name),
        }
    }

    fn determine_severity(&self, pattern: &DataPattern) -> RuleSeverity {
        if pattern.confidence > 0.9 {
            RuleSeverity::Critical
        } else if pattern.confidence > 0.7 {
            RuleSeverity::High
        } else if pattern.confidence > 0.5 {
            RuleSeverity::Medium
        } else {
            RuleSeverity::Low
        }
    }

    fn optimize_condition(&self, condition: &str) -> String {
        // Simple optimization - in practice, this would be more sophisticated
        condition.to_string()
    }
}

impl Agent for ValidationRuleAgent {
    fn name(&self) -> &str {
        "ValidationRuleAgent"
    }

    async fn execute(&self, input: AgentInput) -> Result<AgentOutput, Box<dyn Error + Send + Sync>> {
        let dataset = input.data.get("dataset")
            .and_then(|v| v.as_str())
            .ok_or("Dataset not provided")?;

        let patterns = self.analyze_data_patterns(dataset).await?;
        let rules = self.generate_rules(patterns).await?;
        let optimized_rules = self.optimize_rules(rules).await?;

        // Save rules to repository
        for rule in &optimized_rules {
            self.rule_repository.save(rule.clone()).await?;
        }

        let mut output_data = HashMap::new();
        output_data.insert("rules_generated".to_string(), serde_json::Value::Number(optimized_rules.len().into()));
        output_data.insert("rules".to_string(), serde_json::to_value(optimized_rules)?);

        Ok(AgentOutput {
            success: true,
            data: output_data,
            message: format!("Generated {} validation rules", optimized_rules.len()),
            timestamp: Utc::now(),
        })
    }

    async fn get_status(&self) -> AgentStatus {
        AgentStatus {
            is_active: true,
            last_execution: Some(Utc::now()),
            metrics: HashMap::new(),
            error_count: 0,
        }
    }
}

// A/B Testing Agent
pub struct ABTestingAgent {
    test_repository: ABTestRepository,
    metrics_collector: MetricsCollector,
    statistical_analyzer: StatisticalAnalyzer,
}

impl ABTestingAgent {
    pub fn new(
        test_repository: ABTestRepository,
        metrics_collector: MetricsCollector,
        statistical_analyzer: StatisticalAnalyzer,
    ) -> Self {
        Self {
            test_repository,
            metrics_collector,
            statistical_analyzer,
        }
    }

    pub async fn create_test(&self, rule_a: ValidationRule, rule_b: ValidationRule) -> Result<ABTest, Box<dyn Error + Send + Sync>> {
        let test = ABTest {
            id: format!("test_{}", uuid::Uuid::new_v4()),
            name: format!("A/B Test: {} vs {}", rule_a.name, rule_b.name),
            rule_a,
            rule_b,
            start_date: Utc::now(),
            end_date: None,
            status: TestStatus::Running,
            metrics: TestMetrics {
                rule_a_performance: 0.0,
                rule_b_performance: 0.0,
                statistical_significance: 0.0,
                sample_size: 0,
                confidence_interval: (0.0, 0.0),
            },
        };

        self.test_repository.save(test.clone()).await?;
        Ok(test)
    }

    pub async fn monitor_test(&self, test_id: &str) -> Result<TestMetrics, Box<dyn Error + Send + Sync>> {
        let metrics = self.metrics_collector.collect_metrics(test_id).await?;
        Ok(metrics)
    }

    pub async fn analyze_results(&self, test_id: &str) -> Result<TestResult, Box<dyn Error + Send + Sync>> {
        let result = self.statistical_analyzer.analyze(test_id).await?;
        Ok(result)
    }

    pub async fn promote_winner(&self, test_id: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        let test = self.test_repository.get(test_id).await?;
        let winner = if test.metrics.rule_a_performance > test.metrics.rule_b_performance {
            &test.rule_a
        } else {
            &test.rule_b
        };

        // Promote the winning rule
        self.test_repository.promote_rule(winner).await?;
        Ok(())
    }
}

impl Agent for ABTestingAgent {
    fn name(&self) -> &str {
        "ABTestingAgent"
    }

    async fn execute(&self, input: AgentInput) -> Result<AgentOutput, Box<dyn Error + Send + Sync>> {
        let rule_a_json = input.data.get("rule_a")
            .ok_or("Rule A not provided")?;
        let rule_b_json = input.data.get("rule_b")
            .ok_or("Rule B not provided")?;

        let rule_a: ValidationRule = serde_json::from_value(rule_a_json.clone())?;
        let rule_b: ValidationRule = serde_json::from_value(rule_b_json.clone())?;

        let test = self.create_test(rule_a, rule_b).await?;

        let mut output_data = HashMap::new();
        output_data.insert("test_id".to_string(), serde_json::Value::String(test.id.clone()));
        output_data.insert("test_name".to_string(), serde_json::Value::String(test.name.clone()));

        Ok(AgentOutput {
            success: true,
            data: output_data,
            message: format!("Created A/B test: {}", test.name),
            timestamp: Utc::now(),
        })
    }

    async fn get_status(&self) -> AgentStatus {
        AgentStatus {
            is_active: true,
            last_execution: Some(Utc::now()),
            metrics: HashMap::new(),
            error_count: 0,
        }
    }
}

// Missing Data Agent
pub struct MissingDataAgent {
    data_analyzer: DataCompletenessAnalyzer,
    task_creator: TaskCreator,
    priority_calculator: PriorityCalculator,
}

impl MissingDataAgent {
    pub fn new(
        data_analyzer: DataCompletenessAnalyzer,
        task_creator: TaskCreator,
        priority_calculator: PriorityCalculator,
    ) -> Self {
        Self {
            data_analyzer,
            task_creator,
            priority_calculator,
        }
    }

    pub async fn scan_for_missing_data(&self) -> Result<Vec<MissingDataItem>, Box<dyn Error + Send + Sync>> {
        let missing_items = self.data_analyzer.analyze_completeness().await?;
        Ok(missing_items)
    }

    pub async fn create_tasks(&self, missing_items: Vec<MissingDataItem>) -> Result<Vec<Task>, Box<dyn Error + Send + Sync>> {
        let mut tasks = Vec::new();
        
        for item in missing_items {
            let task = Task {
                id: format!("task_{}", uuid::Uuid::new_v4()),
                title: format!("Complete missing data for {}", item.entity_id),
                description: format!("Missing fields: {}", item.missing_fields.join(", ")),
                priority: item.priority,
                status: TaskStatus::Open,
                assigned_to: None,
                due_date: Some(Utc::now() + chrono::Duration::days(7)),
                created_at: Utc::now(),
                entity_type: item.entity_type,
                entity_id: item.entity_id,
            };
            
            self.task_creator.create_task(task.clone()).await?;
            tasks.push(task);
        }
        
        Ok(tasks)
    }

    pub async fn prioritize_items(&self, items: Vec<MissingDataItem>) -> Result<Vec<MissingDataItem>, Box<dyn Error + Send + Sync>> {
        let prioritized_items = self.priority_calculator.calculate_priorities(items).await?;
        Ok(prioritized_items)
    }
}

impl Agent for MissingDataAgent {
    fn name(&self) -> &str {
        "MissingDataAgent"
    }

    async fn execute(&self, _input: AgentInput) -> Result<AgentOutput, Box<dyn Error + Send + Sync>> {
        let missing_items = self.scan_for_missing_data().await?;
        let prioritized_items = self.prioritize_items(missing_items).await?;
        let tasks = self.create_tasks(prioritized_items).await?;

        let mut output_data = HashMap::new();
        output_data.insert("missing_items_found".to_string(), serde_json::Value::Number(tasks.len().into()));
        output_data.insert("tasks_created".to_string(), serde_json::Value::Number(tasks.len().into()));

        Ok(AgentOutput {
            success: true,
            data: output_data,
            message: format!("Created {} tasks for missing data", tasks.len()),
            timestamp: Utc::now(),
        })
    }

    async fn get_status(&self) -> AgentStatus {
        AgentStatus {
            is_active: true,
            last_execution: Some(Utc::now()),
            metrics: HashMap::new(),
            error_count: 0,
        }
    }
}

// Script Generation Agent
pub struct ScriptGenerationAgent {
    excel_generator: ExcelGenerator,
    template_repository: TemplateRepository,
    validation_integrator: ValidationIntegrator,
}

impl ScriptGenerationAgent {
    pub fn new(
        excel_generator: ExcelGenerator,
        template_repository: TemplateRepository,
        validation_integrator: ValidationIntegrator,
    ) -> Self {
        Self {
            excel_generator,
            template_repository,
            validation_integrator,
        }
    }

    pub async fn generate_script(&self, requirements: ScriptRequirements) -> Result<ExcelScript, Box<dyn Error + Send + Sync>> {
        let template = self.template_repository.get_template(&requirements.entity_type).await?;
        let mut script = ExcelScript {
            id: format!("script_{}", uuid::Uuid::new_v4()),
            name: format!("Update script for {}", requirements.entity_type),
            description: requirements.description,
            template_path: template.path,
            validation_rules: Vec::new(),
            data_mapping: requirements.data_mapping,
            status: ScriptStatus::Draft,
            created_at: Utc::now(),
            approved_at: None,
        };

        // Integrate validation rules
        if let Some(rules) = requirements.validation_rules {
            self.validation_integrator.integrate_rules(&mut script, rules).await?;
        }

        Ok(script)
    }

    pub async fn create_template(&self, entity_type: EntityType) -> Result<String, Box<dyn Error + Send + Sync>> {
        let template_path = self.excel_generator.generate_template(entity_type).await?;
        Ok(template_path)
    }
}

impl Agent for ScriptGenerationAgent {
    fn name(&self) -> &str {
        "ScriptGenerationAgent"
    }

    async fn execute(&self, input: AgentInput) -> Result<AgentOutput, Box<dyn Error + Send + Sync>> {
        let entity_type_str = input.data.get("entity_type")
            .and_then(|v| v.as_str())
            .ok_or("Entity type not provided")?;

        let entity_type = match entity_type_str {
            "Equipment" => EntityType::Equipment,
            "FunctionalLocation" => EntityType::FunctionalLocation,
            "MaintenanceOrder" => EntityType::MaintenanceOrder,
            "SparePart" => EntityType::SparePart,
            _ => return Err("Invalid entity type".into()),
        };

        let requirements = ScriptRequirements {
            entity_type,
            description: input.data.get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("Auto-generated script")
                .to_string(),
            data_mapping: DataMapping {
                source_fields: Vec::new(),
                target_fields: Vec::new(),
                transformation_rules: Vec::new(),
            },
            validation_rules: None,
        };

        let script = self.generate_script(requirements).await?;
        let template_path = self.create_template(entity_type).await?;

        let mut output_data = HashMap::new();
        output_data.insert("script_id".to_string(), serde_json::Value::String(script.id.clone()));
        output_data.insert("template_path".to_string(), serde_json::Value::String(template_path));

        Ok(AgentOutput {
            success: true,
            data: output_data,
            message: format!("Generated script: {}", script.name),
            timestamp: Utc::now(),
        })
    }

    async fn get_status(&self) -> AgentStatus {
        AgentStatus {
            is_active: true,
            last_execution: Some(Utc::now()),
            metrics: HashMap::new(),
            error_count: 0,
        }
    }
}

// AI Agent Hub
pub struct AIAgentHub {
    agents: HashMap<String, Box<dyn Agent>>,
    script_repository: ScriptRepository,
    performance_monitor: PerformanceMonitor,
}

impl AIAgentHub {
    pub fn new(
        script_repository: ScriptRepository,
        performance_monitor: PerformanceMonitor,
    ) -> Self {
        Self {
            agents: HashMap::new(),
            script_repository,
            performance_monitor,
        }
    }

    pub async fn register_agent(&mut self, name: String, agent: Box<dyn Agent>) {
        self.agents.insert(name, agent);
    }

    pub async fn execute_workflow(&self, workflow: Workflow) -> Result<WorkflowResult, Box<dyn Error + Send + Sync>> {
        let mut results = Vec::new();
        
        for step in workflow.steps {
            if let Some(agent) = self.agents.get(&step.name) {
                let input = AgentInput {
                    data: HashMap::new(),
                    parameters: step.parameters,
                };
                
                let output = agent.execute(input).await?;
                results.push(output);
            }
        }
        
        Ok(WorkflowResult {
            workflow_id: workflow.id,
            success: results.iter().all(|r| r.success),
            results,
            timestamp: Utc::now(),
        })
    }

    pub async fn get_scripts(&self) -> Result<Vec<ExcelScript>, Box<dyn Error + Send + Sync>> {
        self.script_repository.get_all().await
    }

    pub async fn approve_script(&self, script_id: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.script_repository.approve(script_id).await
    }
}

// Supporting structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub description: String,
    pub priority: Priority,
    pub status: TaskStatus,
    pub assigned_to: Option<String>,
    pub due_date: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub entity_type: EntityType,
    pub entity_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Open,
    InProgress,
    Completed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptRequirements {
    pub entity_type: EntityType,
    pub description: String,
    pub data_mapping: DataMapping,
    pub validation_rules: Option<Vec<ValidationRule>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub workflow_id: String,
    pub success: bool,
    pub results: Vec<AgentOutput>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub winner: String,
    pub confidence: f64,
    pub recommendation: String,
}

// Trait definitions for dependencies
pub trait RuleGenerationModel: Send + Sync {
    async fn analyze_patterns(&self, dataset: &str) -> Result<Vec<DataPattern>, Box<dyn Error + Send + Sync>>;
}

pub trait DatabaseConnection: Send + Sync {}

pub trait ValidationRuleRepository: Send + Sync {
    async fn save(&self, rule: ValidationRule) -> Result<(), Box<dyn Error + Send + Sync>>;
    async fn get(&self, id: &str) -> Result<ValidationRule, Box<dyn Error + Send + Sync>>;
}

pub trait ABTestRepository: Send + Sync {
    async fn save(&self, test: ABTest) -> Result<(), Box<dyn Error + Send + Sync>>;
    async fn get(&self, id: &str) -> Result<ABTest, Box<dyn Error + Send + Sync>>;
    async fn promote_rule(&self, rule: &ValidationRule) -> Result<(), Box<dyn Error + Send + Sync>>;
}

pub trait MetricsCollector: Send + Sync {
    async fn collect_metrics(&self, test_id: &str) -> Result<TestMetrics, Box<dyn Error + Send + Sync>>;
}

pub trait StatisticalAnalyzer: Send + Sync {
    async fn analyze(&self, test_id: &str) -> Result<TestResult, Box<dyn Error + Send + Sync>>;
}

pub trait DataCompletenessAnalyzer: Send + Sync {
    async fn analyze_completeness(&self) -> Result<Vec<MissingDataItem>, Box<dyn Error + Send + Sync>>;
}

pub trait TaskCreator: Send + Sync {
    async fn create_task(&self, task: Task) -> Result<(), Box<dyn Error + Send + Sync>>;
}

pub trait PriorityCalculator: Send + Sync {
    async fn calculate_priorities(&self, items: Vec<MissingDataItem>) -> Result<Vec<MissingDataItem>, Box<dyn Error + Send + Sync>>;
}

pub trait ExcelGenerator: Send + Sync {
    async fn generate_template(&self, entity_type: EntityType) -> Result<String, Box<dyn Error + Send + Sync>>;
}

pub trait TemplateRepository: Send + Sync {
    async fn get_template(&self, entity_type: &EntityType) -> Result<Template, Box<dyn Error + Send + Sync>>;
}

pub trait ValidationIntegrator: Send + Sync {
    async fn integrate_rules(&self, script: &mut ExcelScript, rules: Vec<ValidationRule>) -> Result<(), Box<dyn Error + Send + Sync>>;
}

pub trait ScriptRepository: Send + Sync {
    async fn get_all(&self) -> Result<Vec<ExcelScript>, Box<dyn Error + Send + Sync>>;
    async fn approve(&self, script_id: &str) -> Result<(), Box<dyn Error + Send + Sync>>;
}

pub trait PerformanceMonitor: Send + Sync {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Template {
    pub id: String,
    pub name: String,
    pub path: String,
    pub entity_type: EntityType,
}

// Error types
#[derive(Debug)]
pub struct AgentError {
    message: String,
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Agent error: {}", self.message)
    }
}

impl Error for AgentError {}

impl From<String> for AgentError {
    fn from(message: String) -> Self {
        AgentError { message }
    }
}

// Python integration
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn ai_agents(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ValidationRuleAgent>()?;
    m.add_class::<ABTestingAgent>()?;
    m.add_class::<MissingDataAgent>()?;
    m.add_class::<ScriptGenerationAgent>()?;
    m.add_class::<AIAgentHub>()?;
    Ok(())
} 