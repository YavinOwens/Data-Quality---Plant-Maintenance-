import os
import types


def test_feature_flag_default_disabled(monkeypatch):
    # By default AI agents should be disabled
    monkeypatch.delenv('AI_AGENTS_ENABLED', raising=False)
    import importlib
    app_module = importlib.import_module('logging-service.app'.replace('-', '_'))
    assert getattr(app_module, 'ai_agent_manager', None) is None


def test_logging_service_ready_endpoint(monkeypatch):
    # Minimal import and readiness path presence
    import importlib
    app_module = importlib.import_module('logging-service.app'.replace('-', '_'))
    assert hasattr(app_module, 'readiness_check')


def test_validation_engine_ready_gauge_imports():
    import importlib
    ve = importlib.import_module('validation-engine.app'.replace('-', '_'))
    assert hasattr(ve, 'SERVICE_READY')


