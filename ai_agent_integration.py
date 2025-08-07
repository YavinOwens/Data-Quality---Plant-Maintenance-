#!/usr/bin/env python3
"""
AI Agent Integration Module
Connects Rust-based AI agents with the existing Flask application
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import os
import sys

# Add the Rust library path
sys.path.append(os.path.join(os.path.dirname(__file__), 'target/release'))

try:
    import ai_agent_workflow
except ImportError:
    print("Warning: Rust AI agent library not found. Using Python fallback.")
    ai_agent_workflow = None

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for AI agents"""
    validation_rule_enabled: bool = True
    ab_testing_enabled: bool = True
    missing_data_enabled: bool = True
    script_generation_enabled: bool = True
    schedule_interval_minutes: int = 30
    max_concurrent_agents: int = 4

class AIAgentManager:
    """Manages all AI agents and their workflows"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agents = {}
        self.workflows = {}
        self.is_running = False
        
        # Initialize Rust agents if available
        if ai_agent_workflow:
            self._initialize_rust_agents()
        else:
            self._initialize_python_agents()
    
    def _initialize_rust_agents(self):
        """Initialize Rust-based AI agents"""
        try:
            # Initialize Rust agents
            self.agents['validation_rule'] = ai_agent_workflow.ValidationRuleAgent()
            self.agents['ab_testing'] = ai_agent_workflow.ABTestingAgent()
            self.agents['missing_data'] = ai_agent_workflow.MissingDataAgent()
            self.agents['script_generation'] = ai_agent_workflow.ScriptGenerationAgent()
            
            # Initialize AI Agent Hub
            self.hub = ai_agent_workflow.AIAgentHub()
            
            logger.info("Rust AI agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Rust agents: {e}")
            self._initialize_python_agents()
    
    def _initialize_python_agents(self):
        """Initialize Python-based AI agents (fallback)"""
        from .python_agents import (
            PythonValidationRuleAgent,
            PythonABTestingAgent,
            PythonMissingDataAgent,
            PythonScriptGenerationAgent
        )
        
        self.agents['validation_rule'] = PythonValidationRuleAgent()
        self.agents['ab_testing'] = PythonABTestingAgent()
        self.agents['missing_data'] = PythonMissingDataAgent()
        self.agents['script_generation'] = PythonScriptGenerationAgent()
        
        logger.info("Python AI agents initialized successfully")
    
    async def start_scheduled_workflows(self):
        """Start scheduled workflow execution"""
        self.is_running = True
        
        while self.is_running:
            try:
                await self._execute_scheduled_workflows()
                await asyncio.sleep(self.config.schedule_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in scheduled workflows: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _execute_scheduled_workflows(self):
        """Execute scheduled workflows"""
        workflows = [
            self._validation_rule_workflow,
            self._missing_data_workflow,
            self._ab_testing_workflow,
        ]
        
        # Execute workflows concurrently
        tasks = []
        for workflow in workflows:
            if self._should_execute_workflow(workflow.__name__):
                tasks.append(asyncio.create_task(workflow()))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Workflow {i} failed: {result}")
    
    def _should_execute_workflow(self, workflow_name: str) -> bool:
        """Determine if a workflow should be executed based on configuration"""
        if workflow_name == "validation_rule_workflow":
            return self.config.validation_rule_enabled
        elif workflow_name == "ab_testing_workflow":
            return self.config.ab_testing_enabled
        elif workflow_name == "missing_data_workflow":
            return self.config.missing_data_enabled
        return True
    
    async def _validation_rule_workflow(self):
        """Validation rule creation workflow"""
        try:
            agent = self.agents['validation_rule']
            
            # Analyze data patterns
            input_data = {
                'dataset': 'equipment_data',
                'analysis_type': 'completeness'
            }
            
            result = await agent.execute(input_data)
            
            if result.success:
                logger.info(f"Validation rule workflow completed: {result.message}")
                
                # Create A/B test for new rules if enabled
                if self.config.ab_testing_enabled:
                    await self._create_ab_test_for_rules(result.data.get('rules', []))
            else:
                logger.error(f"Validation rule workflow failed: {result.message}")
                
        except Exception as e:
            logger.error(f"Validation rule workflow error: {e}")
    
    async def _missing_data_workflow(self):
        """Missing data detection workflow"""
        try:
            agent = self.agents['missing_data']
            
            result = await agent.execute({})
            
            if result.success:
                logger.info(f"Missing data workflow completed: {result.message}")
                
                # Generate scripts for missing data if enabled
                if self.config.script_generation_enabled:
                    await self._generate_scripts_for_missing_data(result.data)
            else:
                logger.error(f"Missing data workflow failed: {result.message}")
                
        except Exception as e:
            logger.error(f"Missing data workflow error: {e}")
    
    async def _ab_testing_workflow(self):
        """A/B testing workflow"""
        try:
            agent = self.agents['ab_testing']
            
            # Get active tests
            active_tests = await self._get_active_tests()
            
            for test in active_tests:
                # Monitor test performance
                metrics = await agent.monitor_test(test['id'])
                
                # Analyze results if test is complete
                if test['status'] == 'completed':
                    result = await agent.analyze_results(test['id'])
                    
                    if result.confidence > 0.9:  # Statistical significance threshold
                        await agent.promote_winner(test['id'])
                        logger.info(f"Promoted winning rule for test {test['id']}")
                
        except Exception as e:
            logger.error(f"A/B testing workflow error: {e}")
    
    async def _create_ab_test_for_rules(self, rules: List[Dict]):
        """Create A/B tests for newly generated rules"""
        try:
            agent = self.agents['ab_testing']
            
            for i in range(0, len(rules) - 1, 2):
                rule_a = rules[i]
                rule_b = rules[i + 1]
                
                test_input = {
                    'rule_a': rule_a,
                    'rule_b': rule_b,
                    'test_duration_days': 7
                }
                
                result = await agent.execute(test_input)
                
                if result.success:
                    logger.info(f"Created A/B test: {result.data.get('test_name')}")
                    
        except Exception as e:
            logger.error(f"Error creating A/B tests: {e}")
    
    async def _generate_scripts_for_missing_data(self, data: Dict):
        """Generate Excel scripts for missing data"""
        try:
            agent = self.agents['script_generation']
            
            missing_items = data.get('missing_items_found', 0)
            
            if missing_items > 0:
                # Generate scripts for different entity types
                entity_types = ['Equipment', 'FunctionalLocation', 'MaintenanceOrder']
                
                for entity_type in entity_types:
                    script_input = {
                        'entity_type': entity_type,
                        'description': f'Update script for {entity_type} missing data',
                        'include_validations': True
                    }
                    
                    result = await agent.execute(script_input)
                    
                    if result.success:
                        logger.info(f"Generated script for {entity_type}: {result.data.get('script_id')}")
                        
        except Exception as e:
            logger.error(f"Error generating scripts: {e}")
    
    async def _get_active_tests(self) -> List[Dict]:
        """Get active A/B tests from database"""
        # This would typically query the database
        # For now, return mock data
        return [
            {
                'id': 'test_001',
                'status': 'running',
                'start_date': datetime.now().isoformat()
            }
        ]
    
    async def execute_workflow(self, workflow_name: str, parameters: Dict = None) -> Dict:
        """Execute a specific workflow"""
        try:
            if workflow_name not in self.agents:
                raise ValueError(f"Unknown workflow: {workflow_name}")
            
            agent = self.agents[workflow_name]
            input_data = parameters or {}
            
            result = await agent.execute(input_data)
            
            return {
                'success': result.success,
                'message': result.message,
                'data': result.data,
                'timestamp': result.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                'success': False,
                'message': str(e),
                'data': {},
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_agent_status(self) -> Dict[str, Dict]:
        """Get status of all agents"""
        status = {}
        
        for name, agent in self.agents.items():
            try:
                agent_status = await agent.get_status()
                status[name] = {
                    'is_active': agent_status.is_active,
                    'last_execution': agent_status.last_execution.isoformat() if agent_status.last_execution else None,
                    'metrics': agent_status.metrics,
                    'error_count': agent_status.error_count
                }
            except Exception as e:
                logger.error(f"Error getting status for agent {name}: {e}")
                status[name] = {
                    'is_active': False,
                    'last_execution': None,
                    'metrics': {},
                    'error_count': 0,
                    'error': str(e)
                }
        
        return status
    
    async def get_generated_scripts(self) -> List[Dict]:
        """Get all generated Excel scripts"""
        try:
            if hasattr(self, 'hub'):
                scripts = await self.hub.get_scripts()
                return [
                    {
                        'id': script.id,
                        'name': script.name,
                        'description': script.description,
                        'status': script.status.value,
                        'created_at': script.created_at.isoformat(),
                        'approved_at': script.approved_at.isoformat() if script.approved_at else None
                    }
                    for script in scripts
                ]
            else:
                # Fallback to database query
                return await self._get_scripts_from_database()
                
        except Exception as e:
            logger.error(f"Error getting scripts: {e}")
            return []
    
    async def approve_script(self, script_id: str) -> bool:
        """Approve a generated script"""
        try:
            if hasattr(self, 'hub'):
                await self.hub.approve_script(script_id)
                return True
            else:
                # Fallback to database update
                return await self._approve_script_in_database(script_id)
                
        except Exception as e:
            logger.error(f"Error approving script {script_id}: {e}")
            return False
    
    async def _get_scripts_from_database(self) -> List[Dict]:
        """Get scripts from database (fallback)"""
        # This would query the database for scripts
        return []
    
    async def _approve_script_in_database(self, script_id: str) -> bool:
        """Approve script in database (fallback)"""
        # This would update the database
        return True
    
    def stop(self):
        """Stop the agent manager"""
        self.is_running = False

# Flask integration
class FlaskAIAgentIntegration:
    """Flask integration for AI agents"""
    
    def __init__(self, app=None):
        self.app = app
        self.agent_manager = None
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the Flask app with AI agent integration"""
        self.app = app
        
        # Initialize agent manager
        config = AgentConfig(
            validation_rule_enabled=app.config.get('AI_AGENT_VALIDATION_RULE_ENABLED', True),
            ab_testing_enabled=app.config.get('AI_AGENT_AB_TESTING_ENABLED', True),
            missing_data_enabled=app.config.get('AI_AGENT_MISSING_DATA_ENABLED', True),
            script_generation_enabled=app.config.get('AI_AGENT_SCRIPT_GENERATION_ENABLED', True),
            schedule_interval_minutes=app.config.get('AI_AGENT_SCHEDULE_INTERVAL', 30),
            max_concurrent_agents=app.config.get('AI_AGENT_MAX_CONCURRENT', 4)
        )
        
        self.agent_manager = AIAgentManager(config)
        
        # Register Flask routes
        self._register_routes()
        
        # Start scheduled workflows
        asyncio.create_task(self.agent_manager.start_scheduled_workflows())
    
    def _register_routes(self):
        """Register Flask routes for AI agent endpoints"""
        from flask import jsonify, request
        
        @self.app.route('/api/ai-agents/status', methods=['GET'])
        def get_agent_status():
            """Get status of all AI agents"""
            try:
                status = asyncio.run(self.agent_manager.get_agent_status())
                return jsonify({
                    'success': True,
                    'data': status,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/ai-agents/workflow/<workflow_name>', methods=['POST'])
        def execute_workflow(workflow_name):
            """Execute a specific workflow"""
            try:
                parameters = request.get_json() or {}
                result = asyncio.run(self.agent_manager.execute_workflow(workflow_name, parameters))
                return jsonify(result)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/ai-agents/scripts', methods=['GET'])
        def get_scripts():
            """Get all generated scripts"""
            try:
                scripts = asyncio.run(self.agent_manager.get_generated_scripts())
                return jsonify({
                    'success': True,
                    'data': scripts,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/ai-agents/scripts/<script_id>/approve', methods=['POST'])
        def approve_script(script_id):
            """Approve a generated script"""
            try:
                success = asyncio.run(self.agent_manager.approve_script(script_id))
                return jsonify({
                    'success': success,
                    'message': 'Script approved successfully' if success else 'Failed to approve script',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500

# Usage example
if __name__ == "__main__":
    # Example usage
    config = AgentConfig()
    manager = AIAgentManager(config)
    
    async def main():
        # Start scheduled workflows
        await manager.start_scheduled_workflows()
    
    asyncio.run(main()) 