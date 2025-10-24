from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, Text, func
from sqlalchemy.orm import declarative_base
from pydantic import BaseModel, Field

Base = declarative_base()


class ExamplePayload(BaseModel):
    phone: str = Field(..., example="6281234567890")
    message: str = Field(..., example="This is a test notification from the gateway.")
    user_id: int = Field(..., example=12345)


class Resource(Base):
    __tablename__ = "resources"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    endpoint = Column(String, nullable=False)
    headers = Column(JSON, nullable=True)
    required_params = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    healthy = Column(Boolean, default=True)
    last_checked = Column(DateTime, nullable=True)
    next_health_check_at = Column(DateTime, nullable=True)
    is_health_check_enabled = Column(Boolean, default=True, nullable=False)
    health_check_test_values = Column(JSON, nullable=True)
    health_check_method = Column(String, default='simple', nullable=False)  # 'simple' or 'actual'
    health_check_interval_min = Column(Integer, default=60, nullable=False)
    health_check_interval_max = Column(Integer, default=90, nullable=False)
    send_failure_email = Column(Boolean, default=False)
    notification_email = Column(Text, nullable=True)
    notification_subject = Column(String, nullable=True)
    notification_template = Column(Text, nullable=True)
    send_failure_webhook = Column(Boolean, default=False)
    failure_webhook_url = Column(String, nullable=True)
    failure_webhook_headers = Column(JSON, nullable=True)
    failure_webhook_payload = Column(JSON, nullable=True)
    latest_health_check_result = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class Endpoint(Base):
    __tablename__ = "endpoints"
    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)
    required_params = Column(JSON, nullable=True)
    is_development_mode = Column(Boolean, default=False, nullable=False)
    dev_hold_tasks = Column(Boolean, default=False, nullable=False)
    dev_rules_enabled = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    dev_values = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class ParameterRule(Base):
    __tablename__ = "parameter_rules"
    id = Column(Integer, primary_key=True)
    endpoint_id = Column(Integer, nullable=False, index=True)
    resource_id = Column(Integer, nullable=False, index=True)
    target_key = Column(String, nullable=False)
    source_type = Column(String, nullable=False)
    source_value = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class DevelopmentRule(Base):
    __tablename__ = "development_rules"
    id = Column(Integer, primary_key=True)
    endpoint_id = Column(Integer, index=True, nullable=False)
    condition_param = Column(String, nullable=True)
    condition_value = Column(String, nullable=True)
    target_param = Column(String, nullable=False)
    override_value = Column(String, nullable=False)


class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True)
    client_name = Column(String, nullable=True)
    ref_id = Column(String, nullable=True, index=True)
    scope = Column(String, nullable=True, index=True)
    endpoint_id = Column(Integer, nullable=False)
    endpoint_path = Column(String, nullable=False)
    source_payload = Column(JSON, nullable=False)
    target_payload = Column(JSON, nullable=True)
    client_id = Column(Integer, nullable=True, index=True)
    resource_id = Column(Integer, nullable=True)
    resource_name = Column(String, nullable=True)
    status = Column(String, default="pending", index=True)
    attempts = Column(Integer, default=0)
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=5, nullable=False)
    delay = Column(Integer, default=5, nullable=False)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class TaskLog(Base):
    __tablename__ = "task_logs"
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, nullable=False, index=True)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class Client(Base):
    __tablename__ = "clients"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    token = Column(String, unique=True, nullable=False, index=True)
    endpoint_id = Column(Integer, nullable=True, index=True)
    is_dev_client = Column(Boolean, default=False, nullable=False)
    dev_hold_tasks = Column(Boolean, default=False, nullable=False)
    dev_rules_enabled = Column(Boolean, default=False, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class ClientDevRule(Base):
    __tablename__ = "client_dev_rules"
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, index=True, nullable=False)
    endpoint_id = Column(Integer, index=True, nullable=True)
    condition_param = Column(String, nullable=True)
    condition_value = Column(String, nullable=True)
    target_param = Column(String, nullable=True)
    override_value = Column(String, nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class HealthCheckLog(Base):
    __tablename__ = "health_check_logs"
    id = Column(Integer, primary_key=True)
    resource_id = Column(Integer, index=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    is_success = Column(Boolean, nullable=False)
    status_code = Column(Integer, nullable=True)
    response_text = Column(Text, nullable=True)
    notification_status = Column(String, nullable=True)
    method = Column(String, nullable=True)


class Setting(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True)
    value = Column(String, nullable=True)


class EndpointResourceLink(Base):
    __tablename__ = "endpoint_resource_links"
    id = Column(Integer, primary_key=True)
    endpoint_id = Column(Integer, index=True, nullable=False)
    resource_id = Column(Integer, index=True, nullable=False)
    sequence = Column(Integer, default=0, nullable=False)


class EndpointClientLink(Base):
    __tablename__ = "endpoint_client_links"
    id = Column(Integer, primary_key=True)
    endpoint_id = Column(Integer, index=True, nullable=False)
    client_id = Column(Integer, index=True, nullable=False)


class TaskAttemptLog(Base):
    __tablename__ = "task_attempt_logs"
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, index=True, nullable=False)
    resource_id = Column(Integer, nullable=False)
    resource_name = Column(String, nullable=False)
    attempt_at = Column(DateTime, server_default=func.now())
    status = Column(String, nullable=False)
    details = Column(Text, nullable=True)


class Log(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, index=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    message = Column(Text, nullable=False)
