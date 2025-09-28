"""
Dispatchapy
"""
import os
import asyncio
import json
import uuid
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, create_model
import requests
from fastapi import FastAPI, Request, Form, HTTPException, Path, Body, Header
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.templating import Jinja2Templates
from fastapi import Depends, Cookie
from typing import Optional
import secrets
from typing import Optional, Dict, Any, List
from datetime import timedelta
import smtplib
from email.mime.text import MIMEText
from jinja2 import Environment, FileSystemLoader
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    JSON,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session, Session
from sqlalchemy import cast, String, or_
from starlette.concurrency import run_in_threadpool

# --- Authentication Configuration ---
GATEWAY_ADMIN_PASSWORD = os.getenv("GATEWAY_ADMIN_PASSWORD", "admin")
SESSION_COOKIE_NAME = "gateway_session_token"

# --- Security Dependency ---
# This function will run before every protected route.
async def verify_session(
    # UPDATED: We now explicitly tell FastAPI the name of the cookie to look for.
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME)
):
    if session_token != GATEWAY_ADMIN_PASSWORD:
        # If the cookie is missing or incorrect, force a redirect to the login page.
        raise HTTPException(
            status_code=307, 
            detail="Not authenticated", 
            headers={"Location": "/login"}
        )


# -------------------------
# Config
# -------------------------
DATA_DIR = "data"
WORKER_COUNT = 2
MAX_RETRIES_DEFAULT = 5
TEMPLATES_DIR = "templates"
DEFAULT_FAILURE_EMAIL_SUBJECT = """Dispatchapy Alert: Resource "{resource_name}" is DOWN"""
DEFAULT_FAILURE_EMAIL_TEMPLATE = """
This is an automated alert from your Dispatchapy Gateway.

The resource '{resource_name}' has been detected as unhealthy.
Endpoint: {resource_endpoint}
Time of Failure: {timestamp_utc} UTC

Dispatchapy will stop sending traffic to this resource until it becomes healthy again.
"""

# -------------------------
# Database
# -------------------------

DB_FILE = os.path.join(DATA_DIR, "dispatchapy.db")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
SQLITE_URL = f"sqlite:///{DB_FILE}"


# -------------------------
# App & jinja
# -------------------------
app = FastAPI(
    title="Documentation",
    version="2.0.0"
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    db = SessionLocal()
    endpoints = db.query(Endpoint).filter(Endpoint.is_active == True).all()
    db.close()

    for endpoint in endpoints:
        if not endpoint.required_params: continue

        example_fields = {param: (str, Field(..., example=f"your_{param}_here")) for param in endpoint.required_params}
        DynamicExampleModel = create_model(f"ExampleFor_{endpoint.path}", **example_fields)

        path_item = {
            "post": {
                "summary": f"Configured Endpoint: {endpoint.description or endpoint.path}",
                "description": f"Handles notifications for `{endpoint.path}`. See schema for required parameters.",
                "tags": ["Endpoints"],
                
                "parameters": [
                    {
                        "name": "X-API-Token",
                        "in": "header",
                        "required": True,
                        "description": "The secret API token for the client.",
                        "schema": { "type": "string" }
                    }
                ],

                "requestBody": {
                    "content": {
                        "application/json": { "schema": DynamicExampleModel.model_json_schema() }
                    },
                    "required": True
                },
                "responses": {
                    "200": {
                        "description": "Successful Response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "example": "queued"},
                                        "task_id": {"type": "integer", "example": 123}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        openapi_schema["paths"][f"/api/{endpoint.path}"] = path_item
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
app.mount("/static", StaticFiles(directory="statics"), name="static")

jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# -------------------------
# DB models
# -------------------------
Base = declarative_base()
engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

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
    is_active = Column(Boolean, default=True, nullable=False)
    dev_values = Column(JSON, nullable=True) # Stores {param: dev_value}
    created_at = Column(DateTime, server_default=func.now())


class ParameterRule(Base):
    __tablename__ = "parameter_rules"
    id = Column(Integer, primary_key=True)
    endpoint_id = Column(Integer, nullable=False, index=True)
    resource_id = Column(Integer, nullable=False, index=True) # <-- ADD THIS
    target_key = Column(String, nullable=False)
    source_type = Column(String, nullable=False)
    source_value = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

class DevelopmentRule(Base):
    __tablename__ = "development_rules"
    id = Column(Integer, primary_key=True)
    endpoint_id = Column(Integer, index=True, nullable=False)
    
    # The 'IF' part of the rule
    condition_param = Column(String, nullable=True) # e.g., "scope", "ref". If NULL, the rule always applies.
    condition_value = Column(String, nullable=True) # e.g., "PO_NOTIF", "TEST_USER"

    # The 'THEN' part of the rule
    target_param = Column(String, nullable=False) # The incoming param to override (e.g., "phone")
    override_value = Column(String, nullable=False) # The new value to use

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
    resource_id = Column(Integer, nullable=True)
    resource_name = Column(String, nullable=True)
    status = Column(String, default="pending", index=True) # pending, processing, success, failed
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
    endpoint_id = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())

class HealthCheckLog(Base):
    __tablename__ = "health_check_logs"
    id = Column(Integer, primary_key=True)
    resource_id = Column(Integer, index=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    is_success = Column(Boolean, nullable=False)
    status_code = Column(Integer, nullable=True)
    response_text = Column(Text, nullable=True)
    notification_status = Column(String, nullable=True) # e.g., "sent", "failed"

class Setting(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True)
    value = Column(String, nullable=True)

class EndpointResourceLink(Base):
    __tablename__ = "endpoint_resource_links"
    id = Column(Integer, primary_key=True)
    endpoint_id = Column(Integer, index=True, nullable=False)
    resource_id = Column(Integer, index=True, nullable=False)
    sequence = Column(Integer, default=0, nullable=False) # Order of execution

class TaskAttemptLog(Base):
    __tablename__ = "task_attempt_logs"
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, index=True, nullable=False)
    resource_id = Column(Integer, nullable=False)
    resource_name = Column(String, nullable=False)
    attempt_at = Column(DateTime, server_default=func.now())
    status = Column(String, nullable=False) # e.g., "success", "failed"
    details = Column(Text, nullable=True) # To store status code or error message

class Log(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, index=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    message = Column(Text, nullable=False)

def generate_token():
    return f"dpy_{secrets.token_urlsafe(32)}"


Base.metadata.create_all(bind=engine)

# -------------------------
# helpers
# -------------------------
def render(name: str, **ctx):
    return HTMLResponse(jinja_env.get_template(name).render(**ctx))

def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_log(db, task_id: int, message: str):
    tl = TaskLog(task_id=task_id, message=message)
    db.add(tl)
    db.commit()

def choose_resource(db, preferred_id: Optional[int] = None) -> Optional[Resource]:
    if preferred_id:
        r = db.query(Resource).filter(Resource.id == preferred_id, Resource.is_active == True).first()
        if r and r.healthy: 
            return r
            
    # Always prioritize resources that are active AND healthy
    r = db.query(Resource).filter(Resource.is_active == True, Resource.healthy == True).order_by(Resource.created_at).first()
    
    # Fallback to any active resource if no healthy ones are found
    return r if r else db.query(Resource).filter(Resource.is_active == True).order_by(Resource.created_at).first()

def build_target_payload(db: Session, endpoint_id: int, resource_id: int, source_payload: Dict[str, Any], task_id: int) -> Dict[str, Any]:
    # 1. Create a mutable copy of the payload to work with.
    working_payload = source_payload.copy()
    endpoint = db.query(Endpoint).get(endpoint_id)
    
    # 2. Apply Development Rule Overrides (if dev mode is enabled)
    if endpoint.is_development_mode:
        dev_rules = db.query(DevelopmentRule).filter(DevelopmentRule.endpoint_id == endpoint.id).all()
        overrides_to_apply = {}

        # UPDATED: This single loop implements "last match wins".
        # As we iterate, later rules that match will overwrite the values from earlier ones.
        for rule in dev_rules:
            apply_rule = False
            # Check for a default rule (no condition)
            if rule.condition_param is None:
                apply_rule = True
            # Check for a conditional rule that matches the payload
            elif working_payload.get(rule.condition_param) == rule.condition_value:
                apply_rule = True
            
            if apply_rule and rule.target_param in working_payload:
                overrides_to_apply[rule.target_param] = rule.override_value
        
        # Now, modify the working payload with the final override values
        if overrides_to_apply:
            add_log(db, task_id, f"DEV MODE: Applying payload overrides: {overrides_to_apply}")
            for param, value in overrides_to_apply.items():
                working_payload[param] = value

    # 3. Perform Standard Parameter Mapping using the (potentially modified) working_payload
    target_payload = {}
    param_rules = db.query(ParameterRule).filter(
        ParameterRule.endpoint_id == endpoint.id,
        ParameterRule.resource_id == resource_id
    ).all()
    
    for rule in param_rules:
        value = None
        if rule.source_type == 'from_source':
            value = working_payload.get(rule.source_value)
        elif rule.source_type == 'generate_uuid':
            value = str(uuid.uuid4())
        
        if value is not None:
            target_payload[rule.target_key] = value

    # 4. Attach the original, unaltered source payload for auditing purposes.
    target_payload["_raw"] = source_payload
    return target_payload

def get_endpoint_status(db, endpoint: Endpoint) -> str:
    # Check for clients
    if db.query(Client).filter(Client.endpoint_id == endpoint.id).count() == 0:
        return "Missing clients"

    # Check for associated resources
    links = db.query(EndpointResourceLink).filter(EndpointResourceLink.endpoint_id == endpoint.id).all()
    if not links:
        return "No resources"

    # Check if all rules are completely mapped
    associated_resource_ids = [link.resource_id for link in links]
    resources = db.query(Resource).filter(Resource.id.in_(associated_resource_ids)).all()
    all_rules = db.query(ParameterRule).filter(ParameterRule.endpoint_id == endpoint.id).all()
    
    for resource in resources:
        if not resource.required_params:
            continue # Skip resources that have no required params
        
        required_params_set = set(resource.required_params)
        
        # Find rules specific to this resource
        rules_for_this_resource = {rule.target_key for rule in all_rules if rule.resource_id == resource.id}
        
        if not required_params_set.issubset(rules_for_this_resource):
            return f"Rules incomplete for '{resource.name}'"

    return "Ready"

def send_failure_notification(db: Session, resource: Resource) -> bool:
    """
    Sends a failure notification email. It now accepts the database session
    as an argument to prevent creating a new, conflicting session.
    """
    if not resource.send_failure_email or not resource.notification_email:
        return False

    # This function now uses the 'db' session that was passed into it.
    smtp_server = get_setting(db, "smtp_server")
    smtp_port = int(get_setting(db, "smtp_port", 587))
    smtp_username = get_setting(db, "smtp_username")
    smtp_password = get_setting(db, "smtp_password")
    email_from = get_setting(db, "email_from")
    smtp_encryption = get_setting(db, "smtp_encryption", "starttls")
    
    if not all([smtp_server, smtp_username, smtp_password, email_from]):
        print("SMTP settings are not fully configured. Cannot send email alert.")
        return False

    recipients = [email.strip() for email in resource.notification_email.split(',')]
    if not recipients:
        return False
    
    template = resource.notification_template or DEFAULT_FAILURE_EMAIL_TEMPLATE
    subject_template = resource.notification_subject or DEFAULT_FAILURE_EMAIL_SUBJECT
    
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    subject = subject_template.format(resource_name=resource.name)
    body = template.format(
        resource_name=resource.name,
        resource_endpoint=resource.endpoint,
        timestamp_utc=timestamp
    )
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = email_from
    msg['To'] = ", ".join(recipients)

    try:
        if smtp_encryption == 'ssl_tls':
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)

        print(f"Sent failure alert for '{resource.name}' to {resource.notification_email}")
        return 'success'
    except Exception as e:
        print(f"Failed to send email alert: {e}")
        return 'failed'
    
def get_setting(db: Session, key: str, default=None):
    """
    Gets a setting from the database using a provided session.
    """
    setting = db.query(Setting).get(key)
    return setting.value if setting else default

def get_healthy_resource(db, endpoint_id: int):
    links = db.query(EndpointResourceLink).filter(
        EndpointResourceLink.endpoint_id == endpoint_id
    ).order_by(EndpointResourceLink.sequence).all()
    
    ordered_resource_ids = [link.resource_id for link in links]
    if not ordered_resource_ids:
        return []

    healthy_resources_query = db.query(Resource).filter(
        Resource.id.in_(ordered_resource_ids),
        Resource.is_active == True,
        Resource.healthy == True
    ).all()
    
    healthy_res_ids = [res.id for res in healthy_resources_query]
    
    # Re-sorts the healthy resources to match the original failover sequence
    return [res_id for res_id in ordered_resource_ids if res_id in healthy_res_ids]

def add_log(db, task_id: int, message: str):
    """Adds a new log entry for a given task."""
    log_entry = Log(task_id=task_id, message=message)
    db.add(log_entry)
    
def perform_health_check(resource: Resource) -> (bool, int, str):
    """
    Performs a health check request and returns the results.
    This function DOES NOT modify the database.
    """
    is_healthy, status_code, response_text = False, None, ""
    try:
        resp = requests.post(
            url=resource.endpoint, 
            headers=resource.headers, 
            json=resource.health_check_test_values or {}, 
            timeout=10
        )
        if 200 <= resp.status_code < 300: 
            is_healthy = True
        status_code, response_text = resp.status_code, resp.text
    except Exception as e:
        response_text = str(e)
    
    return is_healthy, status_code, response_text

def execute_task_attempt(resource: Resource, payload: Dict) -> (bool, int, str):
    """
    Performs a single task execution request and returns the results.
    This function DOES NOT modify the database.
    """
    try:
        resp = requests.post(
            resource.endpoint, headers=resource.headers,
            json=payload, timeout=15
        )
        resp.raise_for_status() # Raise an exception for 4xx/5xx errors
        return True, resp.status_code, resp.text
    except Exception as e:
        # Try to get the status code from the response if it exists
        status_code = getattr(e.response, 'status_code', None)
        return False, status_code, str(e)

def trigger_failure_webhook(resource: Resource):
    if not resource.send_failure_webhook or not resource.failure_webhook_url:
        return True

    try:
        # Populate placeholders in the payload
        payload = resource.failure_webhook_payload or {}
        for key, value in payload.items():
            if isinstance(value, str):
                payload[key] = value.format(
                    resource_name=resource.name,
                    resource_endpoint=resource.endpoint,
                    timestamp_utc=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                )

        requests.post(
            url=resource.failure_webhook_url,
            headers=resource.failure_webhook_headers,
            json=payload,
            timeout=10
        )
        print(f"Successfully triggered failure webhook for '{resource.name}'")
        return 'success'
    except Exception as e:
        print(f"Failed to trigger failure webhook for '{resource.name}': {e}")
        return 'failed'
# -------------------------
# Worker & sending logic
# -------------------------


async def worker_loop(worker_id: str):
    print(f"[{worker_id}] Worker started.")
    worker_attempt = 1
    while True:
        await asyncio.sleep(1) # Poll more frequently for responsiveness
        print(f"Start loop {worker_attempt}-th of [{worker_id}] since startup")
        worker_attempt += 1
        task_id_to_process = None
        db = SessionLocal()
        try:
            # Find and lock a single pending task
            task = db.query(Task).filter(Task.status == "pending").first()
            if task:
                healthy_resources = get_healthy_resource(db, task.endpoint_id)
                if healthy_resources:
                    task_id_to_process = task.id
                    delay = task.delay
                    task.status = "processing"
                    db.commit()
        finally:
            db.close()

        if not task_id_to_process:
            await asyncio.sleep(10)
            continue
        else:
            await asyncio.sleep(delay)
        # --- The Main Logic ---
        # The slow work is handed off to the background thread.
        # The worker does NOT wait for it to finish. It's "fire and forget".
        # This makes the worker loop extremely fast and able to handle many tasks.
        print(f"[{worker_id}] Dispatched task {task_id_to_process} to background thread.")
        await run_in_threadpool(run_task_in_thread, task_id_to_process, healthy_resources)
        

def run_task_in_thread(task_id: int,healthy_resources) -> (bool, int, str):
    """
    This function is designed to be run in a separate thread.
    It creates its own database session to be thread-safe.
    """
    db = SessionLocal()
    try:
        t = db.query(Task).get(task_id)
        if not t:
            return False, None, "Task not found in thread."
        if not healthy_resources:
            # This case means we should just put it back to pending
            t.status = "pending"
            db.commit()
            return False, None, "No healthy resources available."

        final_details = "Task failed on all available resources."
        for resource in healthy_resources:
            resource = db.query(Resource).get(resource)
            if t.retry_count >= t.max_retries:
                break
            t.retry_count += 1
            t.resource_id = resource.id
            t.resource_name = resource.name
            t.target_payload = build_target_payload(db, t.endpoint_id, resource.id, t.source_payload,t.id)
            log = f"Attempt {t.retry_count}/{t.max_retries} task [{t.id}] via '{resource.name}'"
            print(log)
            add_log(db, t.id, log)
            is_success, status_code, details = execute_task_attempt(resource, t.target_payload)
            final_details = details
            attempt_log = TaskAttemptLog(
                task_id=t.id, resource_id=resource.id, resource_name=resource.name,
                status="success" if is_success else "failed",
                details=f"Status Code: {status_code} - Details: {details}"
            )
            db.add(attempt_log)

            if is_success:
                t.status = "success"
                add_log(db, t.id, "Task Succeeded")
                db.commit()
                return True, status_code, "Task Succeeded"
            else:
                db.commit() # Save the failed attempt log and retry_count
        
        # If the loop finishes, all retries have failed
        if t.retry_count >= t.max_retries:
            t.status = "failed"
            t.last_error = f"Max retries reached. Final error: {final_details}"
        else:
            t.status = "pending"
            t.last_error = final_details
            
        add_log(db, t.id, t.last_error)
        db.commit()
        return False, None, t.last_error

    except Exception as e:
        return False, None, f"Critical error in thread: {e}"
    finally:
        db.close()


# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
async def startup():
    print("Performing startup recovery...")
    db = SessionLocal()
    try:
        stuck_tasks = db.query(Task).filter(Task.status == "processing").all()
        if stuck_tasks:
            print(f"Found {len(stuck_tasks)} stuck tasks. Resetting to 'pending'.")
            for task in stuck_tasks:
                task.status = "pending"
                add_log(db, task.id, "System restart: Resetting task status from 'processing' to 'pending'.")
            db.commit()
    finally:
        db.close()
    
    print("Starting background workers...")
    for i in range(WORKER_COUNT):
        asyncio.create_task(worker_loop(f"w-{i}"))
    asyncio.create_task(resource_health_check_loop())

# -------------------------
# Resource health checker
# -------------------------

async def resource_health_check_loop():
    while True:
        await asyncio.sleep(5)
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            query = db.query(Resource).filter(
                Resource.is_active == True, 
                Resource.is_health_check_enabled == True,
                Resource.next_health_check_at <= now
            )
            for r in query.all():
                old_status = r.healthy
                is_healthy, status_code, response_text = perform_health_check(r)

                # Create and save the log entry first
                new_log = HealthCheckLog(
                    resource_id=r.id, is_success=is_healthy, 
                    status_code=status_code, response_text=response_text
                )
                if old_status is True and is_healthy is False:
                    email_sent = send_failure_notification(db, r)
                    notification_status = []
                    if email_sent:
                        notification_status.append(f"email {email_sent}")
                    webhook_sent = trigger_failure_webhook(r)
                    if webhook_sent:
                        notification_status.append(f"webhook {email_sent}")
                    new_log.notification_status = ", ".join(notification_status)
                db.add(new_log)
                db.commit()
                
                # Perform a direct, explicit UPDATE on the Resource table
                db.query(Resource).filter(Resource.id == r.id).update({
                    "healthy": is_healthy,
                    "last_checked": now,
                    "latest_health_check_result": { "success": is_healthy, "status_code": status_code, "detail": response_text[:200] },
                    "next_health_check_at": now + timedelta(seconds=random.randint(r.health_check_interval_min, r.health_check_interval_max))
                })
                db.commit()

                # Prune old logs
                logs_to_keep_ids = [log_id for log_id, in db.query(HealthCheckLog.id).filter(HealthCheckLog.resource_id == r.id).order_by(HealthCheckLog.created_at.desc()).limit(5).all()]
                if len(logs_to_keep_ids) >= 5:
                    db.query(HealthCheckLog).filter(HealthCheckLog.resource_id == r.id, ~HealthCheckLog.id.in_(logs_to_keep_ids)).delete(synchronize_session=False)
                    db.commit()
        finally:
            db.close()

# -------------------------
# UI Routes
# -------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False,dependencies=[Depends(verify_session)])
def dashboard(request: Request):
    db = SessionLocal()
    
    # 1. Fetch only resources that are marked as active
    active_resources = db.query(Resource).filter(Resource.is_active == True).order_by(Resource.name).all()
    
    # 2. Fetch all active endpoints and determine their status
    active_endpoints_query = db.query(Endpoint).filter(Endpoint.is_active == True).order_by(Endpoint.path).all()
    active_endpoints_data = []
    for e in active_endpoints_query:
        active_endpoints_data.append({
            "path": e.path,
            "status": get_endpoint_status(db, e)
        })

    # Fetch recent tasks and stats (this logic is unchanged)
    tasks = db.query(Task).order_by(Task.created_at.desc()).limit(10).all()
    stats = {
        "total": db.query(Task).count(),
        "pending": db.query(Task).filter(Task.status == "pending").count(),
        "success": db.query(Task).filter(Task.status == "success").count(),
        "failed": db.query(Task).filter(Task.status == "failed").count()
    }
    
    context = {
        "request": request,
        "resources": active_resources,
        "endpoints": active_endpoints_data, # Pass the new endpoint data
        "tasks": tasks,
        "stats": stats
    }
    db.close()
    
    return templates.TemplateResponse("dashboard.html", context)

# --- Login and Logout Routes ---
@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", include_in_schema=False)
async def login_submit(request: Request, password: str = Form(...)):
    if password == GATEWAY_ADMIN_PASSWORD:
        response = RedirectResponse(url="/", status_code=303)
        # UPDATED: Added path="/" to make the cookie available site-wide.
        response.set_cookie(
            key=SESSION_COOKIE_NAME, 
            value=GATEWAY_ADMIN_PASSWORD, 
            httponly=True,
            path="/" 
        )
        return response
    else:
        context = {"request": request, "error": "Invalid password. Please try again."}
        return templates.TemplateResponse("login.html", context, status_code=401)
        
@app.post("/logout", include_in_schema=False, dependencies=[Depends(verify_session)])
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key=SESSION_COOKIE_NAME)
    return response

@app.post("/endpoints/add", include_in_schema=False, dependencies=[Depends(verify_session)])
def endpoints_add(
    path: str = Form(...), 
    resource_id: int = Form(...), # Now mandatory, no Optional
    desc: Optional[str] = Form(None), 
    required_params: List[str] = Form([])
):
    path = path.strip().lstrip("/").replace(" ", "_")
    final_params = sorted(list(set(['ref', 'scope'] + required_params)))

    db = SessionLocal()
    e = Endpoint(path=path, resource_id=resource_id, description=desc, required_params=final_params)
    db.add(e); db.commit(); db.refresh(e)
    
    new_endpoint_id = e.id # Get the ID of the new endpoint
    db.close()
    # Redirect to the rules page for the newly created endpoint
    return RedirectResponse(f"/rules/{new_endpoint_id}", status_code=303)

@app.get("/endpoints/{eid}/clients", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def clients_view(request: Request, eid: int):
    db = SessionLocal()
    endpoint = db.query(Endpoint).get(eid)
    if not endpoint:
        raise HTTPException(status_code=404)
    
    clients = db.query(Client).filter(Client.endpoint_id == eid).order_by(Client.name).all()
    
    context = {
        "request": request,
        "endpoint": endpoint,
        "clients": clients
    }
    db.close()
    
    return templates.TemplateResponse("clients.html", context)

@app.post("/endpoints/{eid}/clients/add", include_in_schema=False, dependencies=[Depends(verify_session)])
def clients_add(eid: int, name: str = Form(...)):
    db = SessionLocal()
    new_client = Client(name=name, endpoint_id=eid, token=generate_token())
    db.add(new_client)
    db.commit()
    db.close()
    return RedirectResponse(f"/endpoints/{eid}/clients", status_code=303)

@app.post("/clients/delete/{cid}", include_in_schema=False, dependencies=[Depends(verify_session)])
def clients_delete(cid: int):
    db = SessionLocal()
    client = db.query(Client).get(cid)
    if client:
        endpoint_id = client.endpoint_id
        db.delete(client)
        db.commit()
        db.close()
        return RedirectResponse(f"/endpoints/{endpoint_id}/clients", status_code=303)
    
    db.close()
    # Fallback redirect if client was already deleted
    return RedirectResponse("/endpoints", status_code=303)

@app.post("/endpoints/edit/{eid}/save-all", include_in_schema=False, dependencies=[Depends(verify_session)])
async def endpoint_edit_and_rules_save(eid: int, request: Request, resource_order: List[int] = Form(...)):
    db = SessionLocal()
    form_data = await request.form()
    endpoint = db.query(Endpoint).get(eid)
    if not endpoint:
        raise HTTPException(status_code=404)

    # --- Part 1: Save Endpoint Details (Path, Desc, and Required Params) ---
    endpoint.path = form_data.get("path").strip().lstrip("/").replace(" ", "_")
    endpoint.description = form_data.get("desc")
    
    # ADDED: Get and save the list of required incoming parameters
    required_params = form_data.getlist("required_params")
    endpoint.required_params = sorted(list(set(['ref', 'scope'] + required_params)))

    # --- Part 2: Rebuild Resource Links ---
    db.query(EndpointResourceLink).filter(EndpointResourceLink.endpoint_id == eid).delete()
    for i, resource_id in enumerate(resource_order):
        link = EndpointResourceLink(endpoint_id=eid, resource_id=resource_id, sequence=i)
        db.add(link)

    # --- Part 3: Rebuild All Rules for this Endpoint ---
    db.query(ParameterRule).filter(ParameterRule.endpoint_id == eid).delete()
    for key, value in form_data.items():
        if key.startswith('source_type_'):
            parts = key.split('_')
            resource_id = int(parts[2])
            target_key = "_".join(parts[3:])
            
            source_type = value
            source_value = form_data.get(f'source_value_{resource_id}_{target_key}')
            
            if source_type:
                rule = ParameterRule(
                    endpoint_id=eid, resource_id=resource_id, target_key=target_key,
                    source_type=source_type,
                    source_value=source_value if source_type == 'from_source' else None
                )
                db.add(rule)

    db.commit()
    db.close()
    return RedirectResponse("/endpoints", status_code=303)

@app.get("/endpoints/new", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def new_endpoint_form(request: Request):
    db = SessionLocal()
    
    # NEW: Check for resources before rendering the page
    if db.query(Resource).count() == 0:
        db.close()
        # Redirect the user to the resources page if none exist
        return RedirectResponse(url="/resources", status_code=303)
        
    all_resources_query = db.query(Resource).order_by(Resource.name).all()
    all_resources_json_safe = [{"id": r.id, "name": r.name} for r in all_resources_query]
    
    context = {
        "request": request,
        "all_resources": all_resources_json_safe
    }
    db.close()
    return templates.TemplateResponse("new_endpoint.html", context)

@app.post("/endpoints/new", include_in_schema=False, dependencies=[Depends(verify_session)])
async def new_endpoint_submit(
    request: Request,
    # UPDATED: The form now sends 'resource_sequence'
    resource_sequence: List[int] = Form([]) 
):
    db = SessionLocal()
    form_data = await request.form()
    
    # Create the basic endpoint
    path = form_data.get("path").strip().lstrip("/").replace(" ", "_")
    required_params = form_data.getlist("required_params")
    
    new_endpoint = Endpoint(
        path=path,
        description=form_data.get("desc"),
        required_params=sorted(list(set(['ref', 'scope'] + required_params)))
    )
    db.add(new_endpoint)
    db.commit()
    db.refresh(new_endpoint)

    # Link the selected resources in their specified order
    # UPDATED: The loop now uses the correct 'resource_sequence' variable
    for i, resource_id in enumerate(resource_sequence):
        if resource_id: # Ensure we don't save empty/unassigned selections
            link = EndpointResourceLink(endpoint_id=new_endpoint.id, resource_id=resource_id, sequence=i)
            db.add(link)
    
    db.commit()
    new_id = new_endpoint.id
    db.close()
    
    # Redirect to the rules page as the next step in the workflow
    return RedirectResponse(f"/endpoints/{new_id}/rules?is_new=true", status_code=303)

@app.get("/endpoints/{eid}/rules", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def rules_form(request: Request, eid: int):
    db = SessionLocal()
    endpoint = db.query(Endpoint).get(eid)
    if not endpoint:
        raise HTTPException(status_code=404)

    # --- Fetch Parameter Mapping Data ---
    links = db.query(EndpointResourceLink).filter(EndpointResourceLink.endpoint_id == eid).order_by(EndpointResourceLink.sequence).all()
    
    # --- THIS IS THE FIX ---
    # Ensure the list of resource IDs is unique while preserving the original order.
    # This prevents the template from rendering duplicate resource cards.
    ordered_ids = [link.resource_id for link in links]
    unique_ordered_ids = list(dict.fromkeys(ordered_ids))
    # ----------------------
    
    associated_resources = []
    if unique_ordered_ids: # Use the new unique list
        res_query = db.query(Resource).filter(Resource.id.in_(unique_ordered_ids)).all()
        res_map = {r.id: r for r in res_query}
        # Build the final list of objects, which will now also be unique
        associated_resources = [res_map[rid] for rid in unique_ordered_ids if rid in res_map]

    all_rules = db.query(ParameterRule).filter(ParameterRule.endpoint_id == eid).all()
    rules_map = {}
    for rule in all_rules:
        if rule.resource_id not in rules_map:
            rules_map[rule.resource_id] = {}
        rules_map[rule.resource_id][rule.target_key] = rule

    # --- Fetch Development Rules Data ---
    dev_rules_query = db.query(DevelopmentRule).filter(DevelopmentRule.endpoint_id == eid).all()
    dev_rules_json_safe = [
        {
            "target_param": rule.target_param,
            "condition_param": rule.condition_param,
            "condition_value": rule.condition_value,
            "override_value": rule.override_value
        } for rule in dev_rules_query
    ]

    context = {
        "request": request,
        "endpoint": endpoint,
        "associated_resources": associated_resources, # Pass the corrected, unique list
        "rules_map": rules_map,
        "dev_rules": dev_rules_json_safe
    }
    db.close()
    return templates.TemplateResponse("rules.html", context)

@app.post("/endpoints/{eid}/toggle_dev_mode", include_in_schema=False, dependencies=[Depends(verify_session)])
def toggle_dev_mode(eid: int, request: Request):
    db = SessionLocal()
    endpoint = db.query(Endpoint).get(eid)
    if endpoint:
        endpoint.is_development_mode = not endpoint.is_development_mode
        db.commit()
    db.close()

    # Check the referrer to redirect back to the correct page
    referer = request.headers.get("referer")
    if referer and f"/endpoints/{eid}/rules" in referer:
        # If the toggle was clicked on the rules page, stay there.
        return RedirectResponse(f"/endpoints/{eid}/rules", status_code=303)
    else:
        # Otherwise, go back to the main endpoint list.
        return RedirectResponse("/endpoints", status_code=303)
    
@app.post("/endpoints/{eid}/rules", include_in_schema=False, dependencies=[Depends(verify_session)])
async def rules_save(eid: int, 
    request: Request,
    is_new: bool = Form(False)
    ):
    db = SessionLocal()
    try:
        form_data = await request.form()
        
        # --- Part 1: Save Parameter Mapping Rules ---
        db.query(ParameterRule).filter(ParameterRule.endpoint_id == eid).delete()
        
        for key, value in form_data.items():
            if key.startswith('source_type_'):
                parts = key.split('_')
                resource_id = int(parts[2])
                target_key = "_".join(parts[3:])
                source_type = value
                source_value_key = f'source_value_{resource_id}_{target_key}'
                source_value = form_data.get(source_value_key)
                
                if source_type:
                    rule = ParameterRule(
                        endpoint_id=eid, 
                        resource_id=resource_id, 
                        target_key=target_key,
                        source_type=source_type, 
                        source_value=source_value if source_type == 'from_source' else None
                    )
                    db.add(rule)
        
        # --- Part 2: Save Development Rules ---
        db.query(DevelopmentRule).filter(DevelopmentRule.endpoint_id == eid).delete()
        
        # Retrieve the lists of data submitted by the form
        condition_params = form_data.getlist("dev_condition_param")
        condition_values = form_data.getlist("dev_condition_value")
        target_params = form_data.getlist("dev_target_param")
        override_values = form_data.getlist("dev_override_value")
        
        # Loop through the submitted rules and create new objects
        for i, target_param in enumerate(target_params):
            # Only save the rule if the essential fields have values
            if target_param and override_values[i]:
                dev_rule = DevelopmentRule(
                    endpoint_id=eid,
                    condition_param=condition_params[i] or None, # Handles "Always Apply"
                    condition_value=condition_values[i] or None,
                    target_param=target_param,
                    override_value=override_values[i]
                )
                db.add(dev_rule)

        db.commit()
    finally:
        db.close()
        
    if is_new:
        # If it was a new endpoint, go to the clients page next
        return RedirectResponse(f"/endpoints/{eid}/clients", status_code=303)
    else:
        # Otherwise, go back to the main list
        return RedirectResponse("/endpoints", status_code=303)

@app.post("/endpoints/{eid}/toggle_active", include_in_schema=False, dependencies=[Depends(verify_session)])
def toggle_active(eid: int):
    db = SessionLocal()
    endpoint = db.query(Endpoint).get(eid)
    if endpoint:
        endpoint.is_active = not endpoint.is_active
        db.commit()
    db.close()
    return RedirectResponse("/endpoints", status_code=303)

@app.get("/documentation", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def documentation_page():
    return render("documentation.html")

# Add this new function with your other UI routes
@app.get("/tasks", response_class=HTMLResponse,include_in_schema=False, dependencies=[Depends(verify_session)])
def tasks_list(status: Optional[str] = None, search: Optional[str] = None):
    db = SessionLocal()
    query = db.query(Task)

    if status and status in ["pending", "success", "failed", "processing"]:
        query = query.filter(Task.status == status)
    
    if search:
        # Search within the source_payload JSON object
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                cast(Task.source_payload, String).ilike(search_term),
                Task.ref_id.ilike(search_term),
                Task.scope.ilike(search_term)
            )
        )
    tasks = query.order_by(Task.created_at.desc()).all()
    db.close()
    
    return render("tasks.html", 
                  tasks=tasks, 
                  current_status=status, 
                  current_search=search)


# --- Resources ---
@app.get("/resources", response_class=HTMLResponse,include_in_schema=False, dependencies=[Depends(verify_session)])
def resources_view():
    db = SessionLocal()
    resources = db.query(Resource).order_by(Resource.created_at).all()
    db.close()
    return render("resources.html", resources=resources)

@app.post("/resources/edit/{rid}", include_in_schema=False, dependencies=[Depends(verify_session)])
def resource_edit_submit(
    rid: int, 
    name: str = Form(...), 
    endpoint: str = Form(...), 
    header_keys: List[str] = Form([]),
    header_values: List[str] = Form([]),
    required_params: List[str] = Form([]),
    param_test_values: List[str] = Form([]),
    is_health_check_enabled: bool = Form(False),
    interval_min: int = Form(1),
    interval_max: int = Form(2),
    interval_unit: str = Form("minutes"),
    notification_email: Optional[str] = Form(None),
    notification_subject: Optional[str] = Form(None),
    notification_template: Optional[str] = Form(None),
    send_failure_email: bool = Form(False),
    send_failure_webhook: bool = Form(False),
    failure_webhook_url: Optional[str] = Form(None),
    failure_webhook_headers: Optional[str] = Form(None),
    failure_webhook_payload: Optional[str] = Form(None)
):
    db = SessionLocal()
    r = db.query(Resource).get(rid)
    if not r:
        raise HTTPException(status_code=404, detail="Resource not found")
    
    r.name = name
    r.endpoint = endpoint
    r.headers = {key: value for key, value in zip(header_keys, header_values) if key} or None
    r.required_params = required_params or None
    r.is_health_check_enabled = is_health_check_enabled
    r.health_check_test_values = {key: value for key, value in zip(required_params, param_test_values) if key and value} or None
    
    # These are the crucial lines for saving the interval.
    # They were likely missing or incorrect in your file.
    multiplier = {'seconds': 1, 'minutes': 60, 'hours': 3600}.get(interval_unit, 60)
    r.health_check_interval_min = interval_min * multiplier
    r.health_check_interval_max = interval_max * multiplier
        
    r.notification_email = notification_email
    r.notification_subject = notification_subject or DEFAULT_FAILURE_EMAIL_SUBJECT
    r.notification_template = notification_template or DEFAULT_FAILURE_EMAIL_TEMPLATE
    r.send_failure_email = send_failure_email
    r.send_failure_webhook = send_failure_webhook
    r.failure_webhook_url = failure_webhook_url
    r.failure_webhook_headers = json.loads(failure_webhook_headers) if failure_webhook_headers and failure_webhook_headers.strip() else None
    r.failure_webhook_payload = json.loads(failure_webhook_payload) if failure_webhook_payload and failure_webhook_payload.strip() else None
    
    db.commit()
    db.close()
    return RedirectResponse("/resources", status_code=303)

@app.post("/resources/toggle/{rid}",include_in_schema=False, dependencies=[Depends(verify_session)])
def resources_toggle(rid: int):
    db = SessionLocal(); r = db.query(Resource).get(rid)
    if r: r.is_active = not r.is_active; db.commit()
    db.close(); return RedirectResponse("/resources", status_code=303)

@app.post("/resources/delete/{rid}",include_in_schema=False, dependencies=[Depends(verify_session)])
def resources_delete(rid: int):
    db = SessionLocal(); r = db.query(Resource).get(rid)
    if r: db.delete(r); db.commit()
    db.close(); return RedirectResponse("/resources", status_code=303)

@app.get("/resources/new", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def new_resource_form(request: Request):
    context = {
        "request": request,
        "default_subject": DEFAULT_FAILURE_EMAIL_SUBJECT,
        "default_template": DEFAULT_FAILURE_EMAIL_TEMPLATE.strip()
    }
    return templates.TemplateResponse("new_resource.html", context)

@app.post("/resources/new", include_in_schema=False, dependencies=[Depends(verify_session)])
def new_resource_submit(
    name: str = Form(...), 
    endpoint: str = Form(...), 
    header_keys: List[str] = Form([]),
    header_values: List[str] = Form([]),
    required_params: List[str] = Form([]),
    param_test_values: List[str] = Form([]),
    is_health_check_enabled: bool = Form(False),
    interval_min: int = Form(1),
    interval_max: int = Form(2),
    interval_unit: str = Form("minutes"),
    notification_email: Optional[str] = Form(None),
    notification_subject: Optional[str] = Form(None),
    notification_template: Optional[str] = Form(None),
    # --- ADDED FIELDS ---
    send_failure_email: bool = Form(False),
    send_failure_webhook: bool = Form(False),
    failure_webhook_url: Optional[str] = Form(None),
    failure_webhook_headers: Optional[str] = Form(None),
    failure_webhook_payload: Optional[str] = Form(None)
):
    db = SessionLocal()
    multiplier = {'seconds': 1, 'minutes': 60, 'hours': 3600}.get(interval_unit, 60)
    min_seconds = interval_min * multiplier
    max_seconds = interval_max * multiplier
        
    headers_dict = {key: value for key, value in zip(header_keys, header_values) if key}
    test_values_dict = {key: value for key, value in zip(required_params, param_test_values) if key and value}
    
    r = Resource(
        name=name, 
        endpoint=endpoint, 
        headers=headers_dict or None,
        required_params=required_params or None, 
        is_active=True, 
        healthy=True,
        is_health_check_enabled=is_health_check_enabled,
        health_check_test_values=test_values_dict or None,
        health_check_interval_min=min_seconds,
        health_check_interval_max=max_seconds,
        next_health_check_at=datetime.utcnow(),
        notification_email=notification_email,
        notification_subject=notification_subject or DEFAULT_FAILURE_EMAIL_SUBJECT,
        notification_template=notification_template or DEFAULT_EMAIL_TEMPLATE,
        # --- ADDED LOGIC ---
        send_failure_email=send_failure_email,
        send_failure_webhook=send_failure_webhook,
        failure_webhook_url=failure_webhook_url,
        failure_webhook_headers=json.loads(failure_webhook_headers) if failure_webhook_headers and failure_webhook_headers.strip() else None,
        failure_webhook_payload=json.loads(failure_webhook_payload) if failure_webhook_payload and failure_webhook_payload.strip() else None
    )
    db.add(r)
    db.commit()
    db.close()
    return RedirectResponse("/resources", status_code=303)

@app.get("/resources/edit/{rid}", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def resource_edit_form(request: Request, rid: int):
    db = SessionLocal()
    resource = db.query(Resource).get(rid)
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")
    
    # UPDATED: Convert min/max seconds to a human-readable value and unit
    min_interval = resource.health_check_interval_min
    if min_interval >= 3600 and min_interval % 3600 == 0:
        resource.interval_unit = 'hours'
        resource.interval_min = min_interval // 3600
        resource.interval_max = resource.health_check_interval_max // 3600
    elif min_interval >= 60 and min_interval % 60 == 0:
        resource.interval_unit = 'minutes'
        resource.interval_min = min_interval // 60
        resource.interval_max = resource.health_check_interval_max // 60
    else:
        resource.interval_unit = 'seconds'
        resource.interval_min = min_interval
        resource.interval_max = resource.health_check_interval_max

    context = {
        "request": request,
        "resource": resource,
        "default_subject": DEFAULT_FAILURE_EMAIL_SUBJECT,
        "default_template": DEFAULT_FAILURE_EMAIL_TEMPLATE.strip()
    }
    db.close()
    return templates.TemplateResponse("edit_resource.html", context)

@app.get("/resources/{resource_id}/details", include_in_schema=False, dependencies=[Depends(verify_session)])
def get_resource_details(resource_id: int):
    db = SessionLocal()
    resource = db.query(Resource).get(resource_id)
    db.close()
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")
    return {"id": resource.id, "name": resource.name, "required_params": resource.required_params or []}

@app.get("/resources/{rid}/history", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def resource_history(rid: int):
    db = SessionLocal()
    resource = db.query(Resource).get(rid)
    if not resource: 
        raise HTTPException(status_code=404, detail="Resource not found")
    
    # Fetches the last 5 logs in descending order
    logs = db.query(HealthCheckLog).filter(HealthCheckLog.resource_id == rid).order_by(HealthCheckLog.created_at.desc()).limit(5).all()
    
    db.close()
    return render("health_history.html", resource=resource, logs=logs)

@app.post("/resources/{rid}/check", include_in_schema=False, dependencies=[Depends(verify_session)])
def manual_health_check(rid: int):
    db = SessionLocal()
    try:
        resource = db.query(Resource).get(rid)
        if not resource:
            raise HTTPException(status_code=404, detail="Resource not found")
        
        is_healthy, status_code, response_text = perform_health_check(resource)

        # Create and save the log entry first
        new_log = HealthCheckLog(
            resource_id=resource.id, is_success=is_healthy, 
            status_code=status_code, response_text=response_text
        )
        if not is_healthy:
            email_sent = send_failure_notification(db, resource)
            notification_status = []
            if email_sent:
                notification_status.append(f"email {email_sent}")
            webhook_sent = trigger_failure_webhook(resource)
            if webhook_sent:
                notification_status.append(f"webhook {webhook_sent}")
            new_log.notification_status = ", ".join(notification_status)
        db.add(new_log)
        db.commit()

        # --- THIS IS THE FIX ---
        # Perform a direct, explicit UPDATE on the Resource table.
        # This bypasses any stale session state issues.
        db.query(Resource).filter(Resource.id == rid).update({
            "healthy": is_healthy,
            "last_checked": datetime.utcnow(),
            "latest_health_check_result": { "success": is_healthy, "status_code": status_code, "detail": response_text[:200] },
            "next_health_check_at": datetime.utcnow() + timedelta(seconds=random.randint(resource.health_check_interval_min, resource.health_check_interval_max))
        })
        db.commit()
        # ----------------------

        # Prune old logs
        logs_to_keep_ids = [log_id for log_id, in db.query(HealthCheckLog.id).filter(HealthCheckLog.resource_id == rid).order_by(HealthCheckLog.created_at.desc()).limit(5).all()]
        if len(logs_to_keep_ids) >= 5:
            db.query(HealthCheckLog).filter(HealthCheckLog.resource_id == rid, ~HealthCheckLog.id.in_(logs_to_keep_ids)).delete(synchronize_session=False)
            db.commit()
        
        return { "success": is_healthy, "status_code": status_code, "detail": response_text[:200] }
    finally:
        db.close()


# --- Endpoints ---
@app.get("/endpoints", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def endpoints_view(request: Request):
    db = SessionLocal()
    endpoints = db.query(Endpoint).order_by(Endpoint.created_at).all()
    
    # Get a count of all existing resources to control the button state
    resource_count = db.query(Resource).count()
    
    endpoints_ui = []
    for e in endpoints:
        resource_link_count = db.query(EndpointResourceLink).filter(EndpointResourceLink.endpoint_id == e.id).count()
        endpoints_ui.append({
            "id": e.id, 
            "path": e.path, 
            "resource_info": f"{resource_link_count} resource(s)" if resource_link_count > 0 else "None",
            "status": get_endpoint_status(db, e),
            "is_development_mode": e.is_development_mode,
            "is_active": e.is_active
        })
    
    context = {
        "request": request,
        "endpoints": endpoints_ui,
        "resource_count": resource_count # Pass the count to the template
    }
    db.close()
    return templates.TemplateResponse("endpoints.html", context)

@app.post("/endpoints/add",include_in_schema=False, dependencies=[Depends(verify_session)])
def endpoints_add(path: str = Form(...), resource_id: Optional[int] = Form(None), desc: Optional[str] = Form(None), required_params: Optional[str] = Form(None)):
    path = path.strip().lstrip("/").replace(" ", "_")
    parsed_params = json.loads(required_params) if required_params else None
    db = SessionLocal()
    e = Endpoint(path=path, resource_id=int(resource_id) if resource_id else None, description=desc, required_params=parsed_params)
    db.add(e); db.commit(); db.close()
    return RedirectResponse("/endpoints", status_code=303)

@app.get("/endpoints/edit/{eid}", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def endpoint_edit_form(request: Request, eid: int):
    db = SessionLocal()
    endpoint = db.query(Endpoint).get(eid)
    if not endpoint: raise HTTPException(404)

    associated_links = db.query(EndpointResourceLink).filter(EndpointResourceLink.endpoint_id == eid).order_by(EndpointResourceLink.sequence).all()
    ordered_ids = [link.resource_id for link in associated_links]
    associated_resources_ids = list(dict.fromkeys(ordered_ids))
    
    all_resources_query = db.query(Resource).all()
    res_map = {r.id: r for r in all_resources_query}
    
    associated_resources_objects = [res_map[rid] for rid in associated_resources_ids if rid in res_map]
    
    # Convert to JSON-safe formats
    associated_resources_json_safe = [{"id": r.id, "name": r.name} for r in associated_resources_objects]
    all_resources_json_safe = [{"id": r.id, "name": r.name} for r in all_resources_query]

    context = {
        "request": request,
        "endpoint": endpoint,
        "associated_resources": associated_resources_json_safe,
        "all_resources": all_resources_json_safe # Pass the complete list
    }
    db.close()
    return templates.TemplateResponse("edit_endpoint.html", context)

@app.post("/endpoints/edit/{eid}", include_in_schema=False, dependencies=[Depends(verify_session)])
async def endpoint_edit_save(
    eid: int, 
    request: Request, 
    # The form will now submit a list of selected resource IDs in order
    resource_sequence: List[int] = Form([]) 
):
    db = SessionLocal()
    form_data = await request.form()
    endpoint = db.query(Endpoint).get(eid)
    if not endpoint: raise HTTPException(404)

    # Save Endpoint Details (Path, Desc, Required Params)
    endpoint.path = form_data.get("path").strip().lstrip("/").replace(" ", "_")
    endpoint.description = form_data.get("desc")
    required_params = form_data.getlist("required_params")
    endpoint.required_params = sorted(list(set(['ref', 'scope'] + required_params)))

    # Rebuild Resource Links based on the new sequence
    db.query(EndpointResourceLink).filter(EndpointResourceLink.endpoint_id == eid).delete()
    for i, resource_id in enumerate(resource_sequence):
        if resource_id: # Ensure we don't save empty selections
            link = EndpointResourceLink(endpoint_id=eid, resource_id=resource_id, sequence=i)
            db.add(link)
    
    db.commit()
    db.close()
    return RedirectResponse("/endpoints", status_code=303)

@app.post("/endpoints/delete/{eid}", include_in_schema=False, dependencies=[Depends(verify_session)])
def endpoints_delete(eid: int):
    db = SessionLocal()
    try:
        endpoint_to_delete = db.query(Endpoint).get(eid)
        if endpoint_to_delete:
            
            # --- NEW: Cascading Delete Logic ---
            # Before deleting the endpoint, delete all of its children records
            # to maintain data integrity.
            
            print(f"Deleting endpoint {eid} and all associated data...")

            # Delete all associated clients
            db.query(Client).filter(Client.endpoint_id == eid).delete(synchronize_session=False)

            # Delete all associated parameter mapping rules
            db.query(ParameterRule).filter(ParameterRule.endpoint_id == eid).delete(synchronize_session=False)
            
            # Delete all associated development rules
            db.query(DevelopmentRule).filter(DevelopmentRule.endpoint_id == eid).delete(synchronize_session=False)

            # Delete all resource links associated with this endpoint
            db.query(EndpointResourceLink).filter(EndpointResourceLink.endpoint_id == eid).delete(synchronize_session=False)
            
            # Finally, delete the endpoint itself
            db.delete(endpoint_to_delete)
            
            # Commit all the deletions in a single transaction
            db.commit()
            print("Deletion complete.")

    finally:
        db.close()
    
    return RedirectResponse("/endpoints", status_code=303)

# --- Parameter Rules ---
@app.get("/rules/{endpoint_id}",include_in_schema=False, response_class=HTMLResponse)
def rules_view(endpoint_id: int):
    db = SessionLocal()
    endpoint = db.query(Endpoint).get(endpoint_id)
    if not endpoint:
        raise HTTPException(404)

    # --- NEW LOGIC STARTS HERE ---
    
    # 1. Get the list of required keys from the linked resource
    required_keys = []
    if endpoint.resource_id:
        resource = db.query(Resource).get(endpoint.resource_id)
        if resource and resource.required_params:
            required_keys = resource.required_params

    # 2. Get existing rules and map them by target_key for easy lookup
    existing_rules = db.query(ParameterRule).filter(ParameterRule.endpoint_id == endpoint_id).all()
    existing_rules_map = {rule.target_key: rule for rule in existing_rules}
    # --- NEW LOGIC ENDS HERE ---
    
    db.close()
    
    # Pass the new data to the template
    return render("rules.html", 
                  endpoint=endpoint, 
                  required_keys=required_keys, 
                  existing_rules_map=existing_rules_map)

# This replaces the old /rules/add and /rules/delete routes
@app.post("/rules/{endpoint_id}/save",include_in_schema=False, dependencies=[Depends(verify_session)])
async def rules_save(endpoint_id: int, request: Request):
    db = SessionLocal()
    endpoint = db.query(Endpoint).get(endpoint_id)
    if not endpoint:
        raise HTTPException(404)

    form_data = await request.form()
    
    # Determine the required keys from the linked resource
    required_keys = []
    if endpoint.resource_id:
        resource = db.query(Resource).get(endpoint.resource_id)
        if resource and resource.required_params:
            required_keys = resource.required_params
    
    # Easiest approach: Delete all old rules and recreate them from the form
    db.query(ParameterRule).filter(ParameterRule.endpoint_id == endpoint_id).delete()
    
    # Create new rules based on the submitted form
    for key in required_keys:
        source_type = form_data.get(f"source_type_{key}")
        source_value = form_data.get(f"source_value_{key}")

        if source_type: # Only create a rule if a source type was selected
            new_rule = ParameterRule(
                endpoint_id=endpoint_id,
                target_key=key,
                source_type=source_type,
                source_value=source_value if source_type == 'from_source' else None
            )
            db.add(new_rule)
            
    db.commit()
    db.close()
    return RedirectResponse("/endpoints", status_code=303)

# --- Tasks ---
@app.get("/task/{tid}", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def task_detail(tid: int):
    db = SessionLocal()
    task = db.query(Task).get(tid)
    if not task:
        raise HTTPException(404)
    logs = db.query(Log).filter(Log.task_id == tid).order_by(Log.created_at.asc()).all()
    # NEW: Fetch the attempt logs
    attempt_logs = db.query(TaskAttemptLog).filter(TaskAttemptLog.task_id == tid).order_by(TaskAttemptLog.attempt_at.asc()).all()
    db.close()
    return render("task_detail.html", task=task, logs=logs, attempt_logs=attempt_logs)

@app.post("/task/{tid}/retry", include_in_schema=False, dependencies=[Depends(verify_session)])
def task_retry(tid: int):
    db = SessionLocal()
    try:
        t = db.query(Task).get(tid)
        if not t:
            raise HTTPException(status_code=404, detail="Task not found.")
        if t.status == "success":
            return JSONResponse({"success": False, "message": "Task has already succeeded."}, status_code=400)

        # Reset the task to its original state for a fresh set of retries.
        # This works even if the task had previously reached its max_retries limit.
        t.status = "pending"
        t.retry_count = 0 # This is the crucial reset
        t.last_error = None
        t.resource_id = None
        t.resource_name = None
        
        db.commit()
        add_log(db, t.id, "Manual retry requested. Task state has been reset.")
        return {"success": True, "message": "Task has been re-queued for processing."}
    finally:
        db.close()

# Setting
@app.get("/settings", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def settings_form(request: Request):
    db = SessionLocal()
    settings = {
        "smtp_server": get_setting(db, "smtp_server"),
        "smtp_port": get_setting(db, "smtp_port", "587"),
        "smtp_username": get_setting(db, "smtp_username"),
        "smtp_password": get_setting(db, "smtp_password"),
        "email_from": get_setting(db, "email_from"),
        "smtp_encryption": get_setting(db, "smtp_encryption", "starttls"),
        "max_failure_attempts": get_setting(db, "max_failure_attempts", "5"),
        "task_delay_min": get_setting(db, "task_delay_min", "1"),
        "task_delay_max": get_setting(db, "task_delay_max", "5")
    }
    db.close()
    return templates.TemplateResponse("settings.html", {"request": request, "settings": settings})

@app.post("/settings", include_in_schema=False, dependencies=[Depends(verify_session)])
def settings_save(
    smtp_server: str = Form(""), smtp_port: int = Form(587),
    smtp_username: str = Form(""), smtp_password: str = Form(""),
    email_from: str = Form(""), smtp_encryption: str = Form("starttls"),
    max_failure_attempts: int = Form(5),
    task_delay_min: int = Form(1),
    task_delay_max: int = Form(5)
):
    db = SessionLocal()
    settings_data = {
        "smtp_server": smtp_server, "smtp_port": str(smtp_port),
        "smtp_username": smtp_username, "smtp_password": smtp_password,
        "email_from": email_from, "smtp_encryption": smtp_encryption,
        "max_failure_attempts": str(max_failure_attempts),
        "task_delay_min": str(task_delay_min),
        "task_delay_max": str(task_delay_max)
    }
    for key, value in settings_data.items():
        setting = db.query(Setting).get(key)
        if setting:
            setting.value = value
        else:
            db.add(Setting(key=key, value=value))
    db.commit()
    db.close()
    return RedirectResponse("/", status_code=303)

@app.post("/settings/test_connection", include_in_schema=False, dependencies=[Depends(verify_session)])
def test_smtp_connection(
    smtp_server: str = Form(""), smtp_port: int = Form(587),
    smtp_username: str = Form(""), smtp_password: str = Form(""),
    smtp_encryption: str = Form("starttls")
):
    # This function now receives the settings directly from the form for an instant test.
    if not all([smtp_server, smtp_username, smtp_password]):
        raise HTTPException(status_code=400, detail="SMTP Server, Username, and Password must be provided for a test.")

    try:
        if smtp_encryption == 'ssl_tls':
            with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=10) as server:
                server.login(smtp_username, smtp_password)
        else:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
        
        return {"success": True, "message": "Connection and login successful!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")
    
# -------------------------
# Dynamic API Endpoint
# -------------------------

@app.post(
    "/api/{epath:path}",
    tags=["API"],
    summary="Send Notification via Dynamic Endpoint",
    description="Post a JSON payload to a dynamically configured path. The gateway will queue the task for processing. The task will be executed when a healthy resource becomes available.",
    include_in_schema=False
)
async def api_dynamic(
    epath: str = Path(..., description="The `path` of the configured endpoint, e.g., 'send_alert'"), 
    payload: Dict[str, Any] = Body(...),
    x_api_token: Optional[str] = Header(None, description="The authentication token for the client.")
):
    epath = epath.strip().lstrip("/")
    db = SessionLocal()
    try:
        endpoint = db.query(Endpoint).filter(Endpoint.path == epath).first()
        if not endpoint or not endpoint.is_active:
            raise HTTPException(status_code=404, detail="Endpoint not found")

        if not x_api_token:
            raise HTTPException(status_code=401, detail="API token is missing")

        client = db.query(Client).filter(Client.token == x_api_token, Client.endpoint_id == endpoint.id).first()
        if not client:
            raise HTTPException(status_code=403, detail="Invalid API token for this endpoint")

        ref_id = payload.get("ref")
        scope = payload.get("scope")
        if not ref_id or not scope:
            raise HTTPException(status_code=400, detail="Payload must include 'ref' and 'scope' fields")
        
        if endpoint.required_params:
            for required_key in endpoint.required_params:
                if required_key not in payload:
                    raise HTTPException(status_code=400, detail=f"Missing required parameter: {required_key}")
        
        # Get the global max_retries setting from the database
        max_retries = int(get_setting(db, "max_failure_attempts", 5))
        delay = random.randint(int(get_setting(db, "task_delay_min", 1)), int(get_setting(db, "task_delay_max", 5)))
        # Create the task in a 'pending' state. The worker will assign the resource later.
        t = Task(
            endpoint_id=endpoint.id, 
            endpoint_path=endpoint.path, 
            source_payload=payload,
            client_name=client.name, 
            ref_id=ref_id, 
            scope=scope,
            status="pending", # Always starts as pending
            attempts=0, # Start at -1 so the first check looks for sequence 0
            retry_count=0, # Initialize retry count
            max_retries=max_retries,
            delay=delay,
        )
        db.add(t)
        db.commit()
        db.refresh(t)
        
        add_log(db, t.id, f"API task from client '{client.name}' created and is pending execution.")
            
        return JSONResponse({"status": "queued", "task_id": t.id})
    finally:
        db.close()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting Dispatchapy...")
    print(f"DB file: {DB_FILE}")
    uvicorn.run("dispatchapy:app", host="0.0.0.0", port=8000, reload=True)