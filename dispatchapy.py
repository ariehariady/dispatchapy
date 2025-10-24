"""
Dispatchapy
Copyright (c) 2025 Arie Hariady (ariehariady@gmail.com)

High-availability API dispatch gateway. It is designed to be a single, 
reliable entry point for sending notifications and webhooks through multiple providers. 
With features like automatic failover, health checks, and a full UI for configuration, 
it ensures your critical communications are always delivered.
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
from fastapi import FastAPI, Request, Form, HTTPException, Path, Body, Header, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.templating import Jinja2Templates
from fastapi import Depends, Cookie
from fastapi import Response
from typing import Optional
import secrets
from typing import Optional, Dict, Any, List
from datetime import timedelta
import smtplib
from email.mime.text import MIMEText
from jinja2 import Environment, FileSystemLoader
import re
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
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session, Session
from sqlalchemy import cast, String, or_, text
from starlette.concurrency import run_in_threadpool
import fcntl
import time
from threading import Thread
import socket
from urllib.parse import urlparse

# --- Authentication Configuration ---
GATEWAY_ADMIN_PASSWORD = os.getenv("GATEWAY_ADMIN_PASSWORD", "admin")
SESSION_COOKIE_NAME = "gateway_session_token"
CLIENT_SESSION_COOKIE = "client_session_token"

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


async def verify_client_session(
    client_token: Optional[str] = Cookie(None, alias=CLIENT_SESSION_COOKIE)
):
    """Dependency for client-facing pages: ensures a valid client token cookie is present."""
    if not client_token:
        raise HTTPException(status_code=307, headers={"Location": "/client/login"})
    # lightweight check: ensure token exists in DB
    db = SessionLocal()
    try:
        c = db.query(Client).filter(Client.token == client_token).first()
        if not c:
            raise HTTPException(status_code=307, headers={"Location": "/client/login"})
    finally:
        db.close()


async def get_admin_or_client(
    session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME),
    client_token: Optional[str] = Cookie(None, alias=CLIENT_SESSION_COOKIE)
):
    """Dependency that returns {'role': 'admin'} for admin sessions or
    {'role':'client', 'client': <Client>} for valid client sessions. Raises a
    redirect HTTPException if neither session is valid.
    """
    db = SessionLocal()
    try:
        # Debug: log cookie values for troubleshooting client login flows
        try:
            print(f"get_admin_or_client: session_token={session_token!r}, client_token={client_token!r}")
        except Exception:
            pass
        # Admin has precedence if both cookies present
        if session_token == GATEWAY_ADMIN_PASSWORD:
            return {"role": "admin"}

        if client_token:
            c = db.query(Client).filter(Client.token == client_token).first()
            if c:
                return {"role": "client", "client": c}
            # invalid client token - redirect to client login
            raise HTTPException(status_code=307, headers={"Location": "/client/login"})

        # no valid session found - redirect to admin login by default
        raise HTTPException(status_code=307, headers={"Location": "/login"})
    finally:
        db.close()


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

# --- Custom OpenAPI for user/client ---
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi import Request

@app.get("/openapi-user.json", include_in_schema=False)
async def openapi_for_user(request: Request, session_token: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME), client_token: Optional[str] = Cookie(None, alias=CLIENT_SESSION_COOKIE)):
    db = SessionLocal()
    client = None
    try:
        # Determine user role
        if session_token == GATEWAY_ADMIN_PASSWORD:
            endpoints = db.query(Endpoint).filter(Endpoint.is_active == True).all()
        elif client_token:
            client = db.query(Client).filter(Client.token == client_token).first()
            if not client:
                db.close()
                return JSONResponse({"error": "Invalid client token"}, status_code=401)
            links = db.query(EndpointClientLink).filter(EndpointClientLink.client_id == client.id).all()
            endpoint_ids = [l.endpoint_id for l in links]
            endpoints = db.query(Endpoint).filter(Endpoint.id.in_(endpoint_ids), Endpoint.is_active == True).all() if endpoint_ids else []
        else:
            db.close()
            return JSONResponse({"error": "Not authenticated"}, status_code=401)

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        # Always set OpenAPI version field and remove any Swagger field
        openapi_schema["openapi"] = "3.0.0"
        if "swagger" in openapi_schema:
            del openapi_schema["swagger"]

        # Remove all /api/* paths, then add only the allowed endpoints using the same logic as custom_openapi
        openapi_schema["paths"] = {k: v for k, v in openapi_schema["paths"].items() if not k.startswith("/api/")}
        for endpoint in endpoints:
            if not endpoint.required_params:
                continue
            example_fields = {param: (str, Field(..., example=f"your_{param}_here")) for param in endpoint.required_params}
            DynamicExampleModel = create_model(f"ExampleFor_{endpoint.path}", **example_fields)
            # If we have a client session, include their token as default/example for the header param
            token_default = client.token if client is not None else None

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
                            "schema": { "type": "string", **({"example": token_default} if token_default else {}) }
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

        # If no allowed endpoints, add a dummy operation
        if not any(k.startswith("/api/") for k in openapi_schema["paths"]):
            openapi_schema["paths"]["/api/noop"] = {
                "get": {
                    "summary": "No endpoints available",
                    "description": "This is a placeholder operation because you have no allowed endpoints.",
                    "responses": {
                        "200": {
                            "description": "No endpoints available"
                        }
                    }
                }
            }
        db.close()
        return JSONResponse(openapi_schema)
    except Exception as e:
        db.close()
        return JSONResponse({"error": str(e)}, status_code=500)

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
    try:
        endpoints = db.query(Endpoint).filter(Endpoint.is_active == True).all()
    except OperationalError as oe:
        # Attempt auto-migration then retry once
        print('custom_openapi: OperationalError while querying endpoints, attempting auto-migrations...')
        try:
            run_auto_migrations(engine)
            db.close()
            db = SessionLocal()
            endpoints = db.query(Endpoint).filter(Endpoint.is_active == True).all()
        except Exception as e:
            print('custom_openapi: migration retry failed:', e)
            endpoints = []
    finally:
        try: db.close()
        except Exception: pass
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
from models import (
    Base, ExamplePayload, Resource, Endpoint, ParameterRule,
    DevelopmentRule, Task, TaskLog, Client, ClientDevRule,
    HealthCheckLog, Setting, EndpointResourceLink, EndpointClientLink,
    TaskAttemptLog, Log
)

engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

# In-memory manual-check guard to prevent duplicate near-simultaneous manual checks
manual_check_locks = {}


def _sqlite_column_type(col):
    """Map SQLAlchemy column types to simple SQLite type names."""
    from sqlalchemy import String, Integer, Boolean, DateTime, Text, JSON
    if isinstance(col.type, String):
        return 'TEXT'
    if isinstance(col.type, Text):
        return 'TEXT'
    if isinstance(col.type, DateTime):
        return 'DATETIME'
    if isinstance(col.type, Integer):
        return 'INTEGER'
    if isinstance(col.type, Boolean):
        return 'INTEGER'
    # SQLite has no native JSON type; store as TEXT
    if isinstance(col.type, JSON):
        return 'TEXT'
    # Fallback
    return 'TEXT'


def run_auto_migrations(engine):
    """A minimal, non-destructive auto-migration helper for SQLite.

    It inspects each ORM-mapped table and issues ALTER TABLE ADD COLUMN
    for any Column defined in the model but missing in the DB table.
    Only supports simple column types and ADD COLUMN (no renames or drops).
    """
    conn = engine.connect()
    inspector = None
    try:
        from sqlalchemy import inspect
        inspector = inspect(engine)
    except Exception:
        inspector = None

    # Iterate declarative metadata tables instead of relying on the
    # internal class registry (which may not exist in some SQLAlchemy versions).
    for tablename, table in Base.metadata.tables.items():
        try:
            # Get existing columns from SQLite using inspector or PRAGMA fallback
            existing = set()
            try:
                if inspector:
                    col_info = inspector.get_columns(tablename)
                    existing = {c['name'] for c in col_info}
                else:
                    from sqlalchemy import text
                    res = conn.execute(text(f"PRAGMA table_info('{tablename}')")).fetchall()
                    # PRAGMA returns rows: cid, name, type, notnull, dflt_value, pk
                    existing = {r[1] for r in res}
            except Exception:
                existing = set()

            # Compare against model/table columns
            for col in table.columns:
                # Skip primary key columns (can't sensibly add PKs to existing tables)
                if col.primary_key:
                    continue
                if col.name in existing:
                    continue

                # Build a minimal ALTER TABLE statement
                col_type = _sqlite_column_type(col)
                alter_sql = f'ALTER TABLE "{tablename}" ADD COLUMN "{col.name}" {col_type}'
                try:
                    from sqlalchemy import text
                    conn.execute(text(alter_sql))
                    print(f"Auto-migration: Added column {col.name} to {tablename}")
                    # If we added health_check_method, set defaults for existing rows
                    if col.name == 'health_check_method':
                        try:
                            conn.execute(text(f"UPDATE \"{tablename}\" SET health_check_method = 'simple' WHERE health_check_method IS NULL"))
                            print(f"Auto-migration: Set default 'simple' for existing {tablename}.health_check_method rows")
                        except Exception as ee:
                            print(f"Auto-migration: Failed to set default for {tablename}.health_check_method: {ee}")
                except Exception as e:
                    print(f"Auto-migration failed for {tablename}.{col.name}: {e}")
        except Exception as e:
            print(f"Auto-migration: skipped table {tablename} due to error: {e}")
    conn.close()


def ensure_clients_endpoint_nullable(engine):
    """If the existing clients.endpoint_id column is NOT NULL in SQLite,
    recreate the table with endpoint_id nullable by copying data.
    This is a one-time, safe migration applied at runtime when needed.
    """
    conn = engine.connect()
    try:
        from sqlalchemy import text as _text
        # Some SQLAlchemy connection objects require a text() wrapper for PRAGMA calls
        res = conn.execute(_text("PRAGMA table_info('clients')")).fetchall()
        # PRAGMA columns: cid, name, type, notnull, dflt_value, pk
        endpoint_col = None
        for r in res:
            if r[1] == 'endpoint_id':
                endpoint_col = r
                break
        if not endpoint_col:
            return
        notnull = endpoint_col[3]
        if notnull == 0:
            return

        print('Migrating clients.endpoint_id to be nullable...')
        # Build new table with endpoint_id nullable. Keep other columns similar.
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS clients_new (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                token TEXT UNIQUE NOT NULL,
                endpoint_id INTEGER,
                description TEXT,
                created_at DATETIME
            )
        '''))

        conn.execute(text('''
            INSERT INTO clients_new (id, name, token, endpoint_id, description, created_at)
            SELECT id, name, token, endpoint_id, description, created_at FROM clients;
        '''))

        conn.execute(text('DROP TABLE clients;'))
        conn.execute(text('ALTER TABLE clients_new RENAME TO clients;'))
        print('Migration complete: clients.endpoint_id is now nullable')
    except Exception as e:
        print('Failed to alter clients table nullability:', e)
    finally:
        conn.close()


def _acquire_resource_lock(rid: int):
    """Try to acquire a non-blocking file lock for a specific resource.

    Returns an open file object when lock acquired, or None when another
    process/thread holds the lock.
    """
    lock_dir = os.path.join(DATA_DIR, 'locks')
    try:
        os.makedirs(lock_dir, exist_ok=True)
        lock_path = os.path.join(lock_dir, f'resource_{rid}.lock')
        f = open(lock_path, 'w')
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except IOError:
            f.close()
            return None
    except Exception:
        return None
# ORM and Pydantic models were moved to models.py to avoid duplication.
# See: ./models.py

def generate_token():
    return f"dpy_{secrets.token_urlsafe(32)}"


try:
    # Run lightweight auto-migrations first (non-destructive ALTER TABLE ADD COLUMN)
    run_auto_migrations(engine)
except Exception as e:
    print(f"Auto-migration failed: {e}")

# Ensure historic clients.endpoint_id nullability before creating tables or serving requests
try:
    ensure_clients_endpoint_nullable(engine)
except Exception as e:
    import traceback
    print('ensure_clients_endpoint_nullable failed at startup:', e)
    traceback.print_exc()

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

def build_target_payload(db: Session, endpoint_id: int, resource_id: int, source_payload: Dict[str, Any], task_id: int, client_id: Optional[int] = None) -> Dict[str, Any]:
    # 1. Create a mutable copy of the payload to work with.
    working_payload = source_payload.copy()
    endpoint = db.query(Endpoint).get(endpoint_id)
    # Prefer explicit client_id argument; fallback to a marker in the payload for backwards-compat
    if client_id is None:
        try:
            client_id = source_payload.get('_client_id') if isinstance(source_payload, dict) else None
        except Exception:
            client_id = None
    
    # 2. Apply Development Rule Overrides in order of precedence:
    #    1) Client-level dev rules (if client exists and dev_rules_enabled)
    #    2) Endpoint-level dev rules (if endpoint is in dev mode and dev_rules_enabled)
    overrides_to_apply = {}
    # Helper to evaluate a set of rules against the working_payload
    def _eval_rules_and_collect(rules):
        collected = {}
        for rule in rules:
            apply_rule = False
            if not rule.condition_param:
                apply_rule = True
            else:
                payload_val = working_payload.get(rule.condition_param)
                if payload_val is None:
                    apply_rule = False
                else:
                    if not rule.condition_value:
                        apply_rule = True
                    else:
                        try:
                            if re.search(rule.condition_value, str(payload_val)):
                                apply_rule = True
                        except Exception:
                            if str(rule.condition_value) in str(payload_val):
                                apply_rule = True
            # Apply the override if the rule matches. Allow setting the target
            # even if the incoming payload does not currently contain that key.
            if apply_rule:
                collected[rule.target_param] = rule.override_value
        return collected

    # Apply client-level dev rules first
    client = None
    try:
        if client_id:
            client = db.query(Client).get(client_id)
    except Exception:
        client = None

    if client.dev_rules_enabled:
        try:
            client_rules = db.query(ClientDevRule).filter(ClientDevRule.client_id == client.id, ClientDevRule.active == True).all()
            client_overrides = _eval_rules_and_collect(client_rules)
            if client_overrides:
                overrides_to_apply.update(client_overrides)
                add_log(db, task_id, f"CLIENT DEV: Applying client-level overrides: {client_overrides}")
        except Exception:
            pass
    
    # removed stray debug print
    # Then apply endpoint-level dev rules (they override client rules when both match)
    if endpoint.is_development_mode and endpoint.dev_rules_enabled:
        try:
            dev_rules = db.query(DevelopmentRule).filter(DevelopmentRule.endpoint_id == endpoint.id).all()
            endpoint_overrides = _eval_rules_and_collect(dev_rules)
            if endpoint_overrides:
                # endpoint rules take precedence over client rules
                overrides_to_apply.update(endpoint_overrides)
                add_log(db, task_id, f"ENDPOINT DEV: Applying endpoint-level overrides: {endpoint_overrides}")
        except Exception:
            pass

    # Apply collected overrides to the working payload
    if overrides_to_apply:
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
    # Check for clients (many-to-many via EndpointClientLink)
    if db.query(EndpointClientLink).filter(EndpointClientLink.endpoint_id == endpoint.id).count() == 0:
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


def get_setting_int(db: Session, key: str, default: int):
    """Safely read an integer setting from the DB and return a fallback on error."""
    try:
        val = get_setting(db, key, default)
        # If the stored value is None, return default
        if val is None:
            return int(default)
        return int(val)
    except Exception:
        return int(default)

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
    
def perform_health_check(resource: Resource) -> tuple[bool, int, str]:
    """
    Performs a health check request and returns the results.
    This function DOES NOT modify the database.
    """
    is_healthy, status_code, response_text = False, None, ""

    def _format_value(val, orig_payload):
        # Recursively format strings inside dict/list structures
        if isinstance(val, str):
            s = val
            # Simple replacements
            s = s.replace("{datetime}", datetime.utcnow().isoformat())
            s = s.replace("{resource_name}", resource.name or "")
            s = s.replace("{resource_endpoint}", resource.endpoint or "")

            # {param:KEY} -> look up in orig_payload (if present)
            def _param_repl(m):
                key = m.group(1)
                v = orig_payload.get(key)
                return str(v) if v is not None else ""

            s = re.sub(r"\{param:([^}]+)\}", _param_repl, s)
            return s
        elif isinstance(val, dict):
            return {k: _format_value(v, orig_payload) for k, v in val.items()}
        elif isinstance(val, list):
            return [_format_value(x, orig_payload) for x in val]
        else:
            return val

    def _simple_tcp_check(url: str, timeout: float = 3.0):
        try:
            p = urlparse(url)
            host = p.hostname
            port = p.port
            if not host:
                return False, None, "invalid URL"
            if not port:
                port = 443 if p.scheme == 'https' else 80
            sock = socket.create_connection((host, port), timeout=timeout)
            sock.close()
            # Pure TCP connect succeeded — this is a non-invasive "ping-like" result.
            return True, None, f"tcp:{host}:{port}:connected"
        except Exception as e:
            return False, None, f"tcp_error:{str(e)}"

    try:
        method = getattr(resource, 'health_check_method', 'simple')
        if method == 'simple':
            return _simple_tcp_check(resource.endpoint)

        # actual method: POST formatted payload
        raw_payload = resource.health_check_test_values or {}
        formatted_payload = _format_value(raw_payload, raw_payload)

        resp = requests.post(
            url=resource.endpoint,
            headers=resource.headers,
            json=formatted_payload,
            timeout=10
        )
        status_code = resp.status_code
        response_text = resp.text or ''
        is_healthy = 200 <= status_code < 300
        return is_healthy, status_code, response_text
    except Exception as e:
        return False, None, str(e)

def execute_task_attempt(resource: Resource, payload: Dict) -> tuple[bool, int, str]:
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
        return 'skipped'

    def _format_value(val, orig_payload):
        # Recursively format strings inside dict/list structures
        if isinstance(val, str):
            try:
                return val.format(datetime=datetime.utcnow().isoformat(), resource_name=resource.name, **(orig_payload or {}))
            except Exception:
                return val
        elif isinstance(val, dict):
            return {k: _format_value(v, orig_payload) for k, v in val.items()}
        elif isinstance(val, list):
            return [_format_value(v, orig_payload) for v in val]
        else:
            return val

    try:
        payload = resource.failure_webhook_payload or {}
        formatted_payload = _format_value(payload, payload)
        requests.post(
            url=resource.failure_webhook_url,
            headers=resource.failure_webhook_headers,
            json=formatted_payload,
            timeout=10
        )
        print(f"Successfully triggered failure webhook for '{resource.name}'")
        return 'success'
    except Exception as e:
        print(f"Failed to trigger failure webhook for '{resource.name}': {e}")
        return 'failed'


async def task_cleanup_loop():
    """
    Background loop that periodically cleans up old tasks based on settings.
    """
    while True:
        await asyncio.sleep(3600)  # Check every hour
        
        db = SessionLocal()
        try:
            # Check if cleanup is enabled
            cleanup_enabled = get_setting(db, "enable_task_cleanup", "false")
            if cleanup_enabled.lower() != "true":
                continue
                
            # Get the cleanup age in days
            cleanup_days = int(get_setting(db, "task_cleanup_days", "30"))
            cutoff_date = datetime.utcnow() - timedelta(days=cleanup_days)
            
            # Find old completed tasks (success or failed status)
            old_tasks_query = db.query(Task).filter(
                Task.created_at < cutoff_date,
                Task.status.in_(["success", "failed"])
            )
            
            old_task_ids = [task.id for task in old_tasks_query.all()]
            if old_task_ids:
                # Delete related logs first (foreign key constraint)
                db.query(Log).filter(Log.task_id.in_(old_task_ids)).delete(synchronize_session=False)
                db.query(TaskLog).filter(TaskLog.task_id.in_(old_task_ids)).delete(synchronize_session=False)
                db.query(TaskAttemptLog).filter(TaskAttemptLog.task_id.in_(old_task_ids)).delete(synchronize_session=False)
                
                # Delete the tasks themselves
                deleted_count = old_tasks_query.delete(synchronize_session=False)
                db.commit()
                
                print(f"Task cleanup: Deleted {deleted_count} old tasks and their logs (older than {cleanup_days} days)")
            
        except Exception as e:
            print(f"Error during task cleanup: {e}")
        finally:
            db.close()


def run_task_in_thread(task_id: int,healthy_resources) -> tuple[bool, int, str]:
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
            # Pass client_id through so client-level dev rules can be applied
            t.target_payload = build_target_payload(db, t.endpoint_id, resource.id, t.source_payload, t.id, client_id=t.client_id)
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


# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
async def startup():
    print("Performing startup recovery...")
    # Ensure DB schema is up-to-date before doing any recovery work
    try:
        run_auto_migrations(engine)
    except Exception as e:
        print('Startup auto-migration failed:', e)

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
    asyncio.create_task(task_cleanup_loop())

# -------------------------
# Resource health checker
# -------------------------

async def resource_health_check_loop():
    while True:
        await asyncio.sleep(5)
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            # Attempt to query resources due for health check. If the DB schema is
            # missing columns (e.g., after a code change), run the auto-migration
            # helper and retry once.
            try:
                query = db.query(Resource).filter(
                    Resource.is_active == True, 
                    Resource.is_health_check_enabled == True,
                    Resource.next_health_check_at <= now
                )
            except OperationalError as oe:
                msg = str(oe).lower()
                if 'no such column' in msg or 'has no column' in msg:
                    print('OperationalError during health-check query, attempting auto-migrations...')
                    try:
                        run_auto_migrations(engine)
                        # Close and reopen session to refresh metadata
                        db.close()
                        db = SessionLocal()
                        query = db.query(Resource).filter(
                            Resource.is_active == True, 
                            Resource.is_health_check_enabled == True,
                            Resource.next_health_check_at <= now
                        )
                    except Exception as mig_e:
                        print(f'Auto-migration retry failed: {mig_e}')
                        continue
                else:
                    # Unknown operational error, re-raise
                    raise
            for r in query.all():
                # Prevent concurrent checks for the same resource using a file lock
                lock = _acquire_resource_lock(r.id)
                if not lock:
                    # Skip this resource this cycle because another check is running
                    continue
                try:
                    old_status = r.healthy

                    # Use the centralized perform_health_check which implements a
                    # non-invasive 'simple' probe (TCP + OPTIONS/HEAD) and an
                    # 'actual' POST-based probe depending on resource config.
                    try:
                        is_healthy, status_code, response_text = perform_health_check(r)
                    except Exception as e:
                        is_healthy = False
                        status_code, response_text = None, str(e)

                    # Create and save the log entry first
                    new_log = HealthCheckLog(
                        resource_id=r.id, is_success=is_healthy,
                        status_code=status_code, response_text=response_text,
                        method=(r.health_check_method or 'simple')
                    )
                    if old_status is True and is_healthy is False:
                        email_sent = send_failure_notification(db, r)
                        notification_status = []
                        if email_sent:
                            notification_status.append(f"email {email_sent}")
                        webhook_sent = trigger_failure_webhook(r)
                        if webhook_sent:
                            notification_status.append(f"webhook {webhook_sent}")
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
                    try:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                        lock.close()
                    except Exception:
                        pass
        finally:
            db.close()

# -------------------------
# UI Routes
# -------------------------

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
            path="/",
            samesite='lax'
        )
        return response
    else:
        context = {"request": request, "error": "Invalid password. Please try again."}
        return templates.TemplateResponse("login.html", context, status_code=401)
        
@app.post("/logout", include_in_schema=False)
def logout(dep=Depends(get_admin_or_client)):
    """Logout endpoint that works for both admin and client sessions.

    - If the request is from an admin session, remove the admin session cookie.
    - If the request is from a client session, remove the client session cookie.
    In both cases, redirect to /login.
    """
    response = RedirectResponse(url="/login", status_code=303)
    try:
        if isinstance(dep, dict) and dep.get('role') == 'admin':
            response.delete_cookie(key=SESSION_COOKIE_NAME, path='/')
        elif isinstance(dep, dict) and dep.get('role') == 'client':
            response.delete_cookie(key=CLIENT_SESSION_COOKIE, path='/')
        else:
            # Fallback: clear both cookies to be safe
            response.delete_cookie(key=SESSION_COOKIE_NAME, path='/')
            response.delete_cookie(key=CLIENT_SESSION_COOKIE, path='/')
    except Exception:
        # Best-effort cookie deletion; continue to redirect regardless
        response.delete_cookie(key=SESSION_COOKIE_NAME, path='/')
        response.delete_cookie(key=CLIENT_SESSION_COOKIE, path='/')
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
    
    # Clients linked via the association table (EndpointClientLink)
    linked_rows = db.query(EndpointClientLink).filter(EndpointClientLink.endpoint_id == eid).all()
    linked_client_ids = [r.client_id for r in linked_rows] if linked_rows else []

    if linked_client_ids:
        associated_clients = db.query(Client).filter(Client.id.in_(linked_client_ids)).order_by(Client.name).all()
        available_clients = db.query(Client).filter(~Client.id.in_(linked_client_ids)).order_by(Client.name).all()
        # Serialize associated clients for safe JSON embedding in templates
        associated_clients_serialized = [ {"id": c.id, "name": c.name, "token": c.token or ""} for c in associated_clients ]
    else:
        associated_clients = []
        available_clients = db.query(Client).order_by(Client.name).all()

    all_clients = db.query(Client).order_by(Client.name).all()
    context = {
        "request": request,
        "endpoint": endpoint,
        # pass both the model list (if needed) and a JSON-serializable copy for templates
        "associated_clients": associated_clients_serialized if linked_client_ids else [],
        "_associated_clients_models": associated_clients,
        "available_clients": available_clients,
        "all_clients": all_clients,
        # keep backward-compatible 'clients' for other parts of the template
        "clients": all_clients
    }
    # Propagate optional error from query params so template can alert it
    qerr = request.query_params.get('error')
    if qerr:
        context['error'] = qerr
    db.close()
    
    return templates.TemplateResponse("clients.html", context)

@app.post("/endpoints/{eid}/clients/add", include_in_schema=False, dependencies=[Depends(verify_session)])
async def clients_add(eid: int, request: Request):
    db = SessionLocal()
    form = await request.form()
    # Read submitted client IDs (may be empty) and optional new client name
    client_ids = form.getlist('client_ids') if hasattr(form, 'getlist') else []
    name = form.get('name')

    try:
        # Replace existing links: remove all links for this endpoint, then add the submitted ones
        db.query(EndpointClientLink).filter(EndpointClientLink.endpoint_id == eid).delete()

        # Add links from submitted client_ids
        for cid in client_ids:
            try:
                cid_i = int(cid)
            except Exception:
                continue
            db.add(EndpointClientLink(endpoint_id=eid, client_id=cid_i))

        # If a new client name was provided, create that client and link it as well
        if name:
            existing = db.query(Client).filter(Client.name == name).first()
            if existing:
                # If already exists, just link it
                db.add(EndpointClientLink(endpoint_id=eid, client_id=existing.id))
            else:
                new_client = Client(name=name, token=generate_token(), endpoint_id=None)
                db.add(new_client)
                db.commit()
                db.refresh(new_client)
                db.add(EndpointClientLink(endpoint_id=eid, client_id=new_client.id))

        db.commit()
    finally:
        db.close()

    return RedirectResponse(f"/endpoints", status_code=303)


@app.get("/clients", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def clients_list(request: Request):
    db = SessionLocal()
    clients = db.query(Client).order_by(Client.name).all()
    # Build mapping of client_id -> list of endpoint paths
    client_endpoints = {}
    links = db.query(EndpointClientLink).all()
    endpoint_ids = list({l.endpoint_id for l in links})
    endpoints = {e.id: e for e in db.query(Endpoint).filter(Endpoint.id.in_(endpoint_ids)).all()} if endpoint_ids else {}
    for l in links:
        client_endpoints.setdefault(l.client_id, []).append(endpoints.get(l.endpoint_id).path if endpoints.get(l.endpoint_id) else str(l.endpoint_id))

    context = {
        "request": request,
        "clients": clients,
        "client_endpoints": client_endpoints
    }
    db.close()
    return templates.TemplateResponse("clients_list.html", context)


@app.get("/endpoints", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def endpoints_list(request: Request):
    """Render the endpoints list page (admin only)."""
    db = SessionLocal()
    try:
        eps = db.query(Endpoint).order_by(Endpoint.path).all()
        endpoints = []
        for e in eps:
            # resource info: comma-separated names of associated resources
            links = db.query(EndpointResourceLink).filter(EndpointResourceLink.endpoint_id == e.id).order_by(EndpointResourceLink.sequence).all()
            res_names = []
            for l in links:
                r = db.query(Resource).get(l.resource_id)
                if r:
                    res_names.append(r.name)
            resource_info = ', '.join(res_names) if res_names else '—'
            status = get_endpoint_status(db, e)
            endpoints.append({
                'id': e.id,
                'path': e.path,
                'resource_info': resource_info,
                'status': status,
                'is_development_mode': bool(getattr(e, 'is_development_mode', False)),
                'is_active': bool(getattr(e, 'is_active', False))
            })

        resource_count = db.query(Resource).count()
        context = {
            'request': request,
            'endpoints': endpoints,
            'resource_count': resource_count
        }
        return templates.TemplateResponse('endpoints.html', context)
    finally:
        db.close()


@app.post('/clients/save', include_in_schema=False)
async def client_save(request: Request, dep=Depends(get_admin_or_client)):
    form_data = await request.form()
    db = SessionLocal()
    try:
        # Ensure historic DB schema (clients.endpoint_id) is nullable to avoid IntegrityError
        try:
            ensure_clients_endpoint_nullable(engine)
        except Exception:
            pass

        cid = form_data.get('id')
        name = form_data.get('name')
        desc = form_data.get('description')
        token = form_data.get('token') or generate_token()
        dev_hold_tasks = True if form_data.get('dev_hold_tasks') in ('1', 'on', 'true', True) else False
        dev_rules_enabled = True if form_data.get('dev_rules_enabled') in ('1', 'on', 'true', True) else False
        endpoint_ids = form_data.getlist('endpoint_ids') if hasattr(form_data, 'getlist') else []

        # Parse submitted client dev rules early (for validation) but do not persist yet
        dev_rules_raw = form_data.get('client_dev_rules_json') if hasattr(form_data, 'get') else None
        parsed_rules = []
        if dev_rules_raw:
            try:
                import json
                parsed_rules = json.loads(dev_rules_raw) or []
            except Exception:
                parsed_rules = []

        # If this request is from a client session, restrict behavior:
        # - clients may only edit their own record (must supply id)
        # - clients may only update development-related settings and description
        #   (name, token and endpoint associations remain admin-only / readonly)
        # Validation: if dev options are requested (dev_hold_tasks or dev_rules_enabled), ensure at least
        # one of: dev_hold_tasks, dev_rules_enabled, or at least one dev rule exists
        if (form_data.get('dev_rules_enabled') in ('1', 'on', 'true', True)) or (form_data.get('dev_hold_tasks') in ('1', 'on', 'true', True)):
            has_rules = bool(parsed_rules)
            # If editing an existing client and no rules were submitted, check existing DB rules
            if cid and not has_rules:
                try:
                    existing_count = db.query(ClientDevRule).filter(ClientDevRule.client_id == int(cid)).count()
                    if existing_count > 0:
                        has_rules = True
                except Exception:
                    has_rules = has_rules

            if not (dev_hold_tasks or dev_rules_enabled or has_rules):
                # Re-render the client form with an error message and bootstrap data preserved
                all_endpoints = db.query(Endpoint).order_by(Endpoint.path).all()
                # linked_endpoint_ids: prefer posted endpoint_ids, fall back to existing links when editing
                linked = []
                try:
                    if endpoint_ids:
                        linked = [int(x) for x in endpoint_ids]
                    elif cid:
                        linked = [l.endpoint_id for l in db.query(EndpointClientLink).filter(EndpointClientLink.client_id == int(cid)).all()]
                except Exception:
                    linked = []

                # compute available rule keys from linked endpoints
                available_rule_keys = set()
                for e in all_endpoints:
                    if linked and e.id in linked and getattr(e, 'required_params', None):
                        for k in e.required_params:
                            available_rule_keys.add(k)
                available_rule_keys.update(['scope', 'ref'])

                context = {
                    'request': request,
                    'client': db.query(Client).get(int(cid)) if cid else None,
                    'generated_token': token,
                    'all_endpoints': [{'id': e.id, 'path': e.path} for e in all_endpoints],
                    'linked_endpoint_ids': linked,
                    'available_rule_keys': sorted(list(available_rule_keys)),
                    'client_dev_rules': parsed_rules,
                    'error': 'Activating Development Mode requires either "Hold incoming tasks" or at least one Dev Rule.'
                }
                db.close()
                return templates.TemplateResponse('client_form.html', context)

        # Unique client name validation on create
        if not cid:
            existing = db.query(Client).filter(Client.name == name).first()
            if existing:
                db.close()
                all_endpoints = db.query(Endpoint).order_by(Endpoint.path).all()
                context = {
                    'request': request,
                    'client': None,
                    'generated_token': token,
                    'all_endpoints': [{'id': e.id, 'path': e.path} for e in all_endpoints],
                    'linked_endpoint_ids': [],
                    'error': f"Client with name '{name}' already exists."
                }
                return templates.TemplateResponse('client_form.html', context)

        if cid:
            client = db.query(Client).get(int(cid))
            if not client:
                raise HTTPException(404)
            if dep.get('role') != 'client':
                client.name = name
                client.description = desc
                client.token = token
            client.dev_hold_tasks = dev_hold_tasks
            client.dev_rules_enabled = dev_rules_enabled
            db.commit()
            client_id = client.id
        else:
            new_client = Client(name=name.strip(), token=token, endpoint_id=None, description=desc, dev_hold_tasks=dev_hold_tasks, dev_rules_enabled=dev_rules_enabled)
            db.add(new_client)
            db.commit()
            db.refresh(new_client)
            client_id = new_client.id

        # Update EndpointClientLink associations
        db.query(EndpointClientLink).filter(EndpointClientLink.client_id == client_id).delete()
        inserted_links = 0
        for eid in endpoint_ids:
            try:
                eid_int = int(eid)
            except Exception:
                continue
            db.add(EndpointClientLink(endpoint_id=eid_int, client_id=client_id))
            inserted_links += 1
        db.commit()
        print(f'Inserted {inserted_links} endpoint links for client_id={client_id}')

        # Persist client-level dev rules if provided (JSON blob)
        try:
            dev_rules_raw = form_data.get('client_dev_rules_json') if hasattr(form_data, 'get') else None
            if dev_rules_raw:
                import json
                try:
                    parsed = json.loads(dev_rules_raw)
                    # delete existing rules for this client and insert new ones
                    db.query(ClientDevRule).filter(ClientDevRule.client_id == client_id).delete()
                    inserted = 0
                    for r in parsed:
                        try:
                            cr = ClientDevRule(
                                client_id=client_id,
                                endpoint_id=r.get('endpoint_id') if r.get('endpoint_id') else None,
                                condition_param=r.get('condition_param'),
                                condition_value=r.get('condition_value'),
                                target_param=r.get('target_param'),
                                override_value=r.get('override_value'),
                                active=bool(r.get('active', True))
                            )
                            db.add(cr)
                            inserted += 1
                        except Exception:
                            continue
                    print(f'Parsed and inserting {inserted} client dev rules for client_id={client_id}')
                except Exception:
                    print('Failed to parse client_dev_rules_json', dev_rules_raw)
        except Exception:
            pass
        db.commit()

    finally:
        db.close()

    # If the form included a return_to field (used when creating from an endpoint), honor it
    try:
        form_return = form_data.get('return_to') if 'form_data' in locals() else None
    except Exception:
        form_return = None

    return RedirectResponse('/' if dep.get('role') == 'client' else form_return or '/clients', status_code=303)


@app.get('/clients/new', response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def client_new_form(request: Request):
    db = SessionLocal()
    try:
        all_endpoints = db.query(Endpoint).order_by(Endpoint.path).all()
        # if caller provided an endpoint_id in query params (e.g. coming from /endpoints/<id>/clients),
        # prefill that endpoint as linked for the new client
        try:
            endpoint_id = request.query_params.get('endpoint_id')
            linked = []
            if endpoint_id:
                try:
                    linked = [int(endpoint_id)]
                except Exception:
                    linked = []
        except Exception:
            linked = []
        # compute available rule keys (union of required_params for linked endpoints)
        available_rule_keys = set()
        for e in all_endpoints:
            if linked and e.id in linked and getattr(e, 'required_params', None):
                for k in e.required_params:
                    available_rule_keys.add(k)
        available_rule_keys.update(['scope', 'ref'])

        context = {
            'request': request,
            'client': None,
            'generated_token': generate_token(),
            'all_endpoints': [{'id': e.id, 'path': e.path} for e in all_endpoints],
            'linked_endpoint_ids': linked,
            'available_rule_keys': sorted(list(available_rule_keys)),
            'client_dev_rules': []
        }
        return templates.TemplateResponse('client_form.html', context)
    finally:
        db.close()


@app.get('/clients/edit/{cid}', response_class=HTMLResponse, include_in_schema=False)
def client_edit_form(request: Request, cid: int, dep=Depends(get_admin_or_client)):
    db = SessionLocal()
    try:
        # If caller is a client session, ensure they can only view/edit their own record
        if isinstance(dep, dict) and dep.get('role') == 'client':
            caller_client = dep.get('client')
            if not caller_client or caller_client.id != cid:
                raise HTTPException(status_code=403)
        client = db.query(Client).get(cid)
        if not client:
            raise HTTPException(404)
        all_endpoints = db.query(Endpoint).order_by(Endpoint.path).all()
        linked_ids = [l.endpoint_id for l in db.query(EndpointClientLink).filter(EndpointClientLink.client_id == cid).all()]
        available_rule_keys = set()
        for e in all_endpoints:
            if e.id in linked_ids and getattr(e, 'required_params', None):
                for k in e.required_params:
                    available_rule_keys.add(k)
        available_rule_keys.update(['scope', 'ref'])

        # fetch existing client dev rules
        client_rules = db.query(ClientDevRule).filter(ClientDevRule.client_id == cid).order_by(ClientDevRule.created_at).all()
        client_rules_json = [
            {
                'id': r.id,
                'endpoint_id': r.endpoint_id,
                'condition_param': r.condition_param,
                'condition_value': r.condition_value,
                'target_param': r.target_param,
                'override_value': r.override_value,
                'active': bool(r.active)
            } for r in client_rules
        ]

        context = {
            'request': request,
            'client': client,
            'is_client': True if isinstance(dep, dict) and dep.get('role') == 'client' else False,
            'generated_token': client.token,
            'all_endpoints': [{'id': e.id, 'path': e.path} for e in all_endpoints],
            'linked_endpoint_ids': linked_ids,
            'available_rule_keys': sorted(list(available_rule_keys)),
            'client_dev_rules': client_rules_json,
            'dev_rules_enabled': client.dev_rules_enabled,
            'dev_hold_tasks': client.dev_hold_tasks
        }
        # If the caller is a client session, render the client-facing settings template
        # if isinstance(dep, dict) and dep.get('role') == 'client':
        #     return templates.TemplateResponse('client_setting.html', context)
        # Admin users receive the full client form
        return templates.TemplateResponse('client_form.html', context)
    finally:
        db.close()




@app.post("/clients/delete/{cid}", include_in_schema=False, dependencies=[Depends(verify_session)])
def clients_delete(cid: int):
    db = SessionLocal()
    client = db.query(Client).get(cid)
    if client:
        # Remove any association links first
        db.query(EndpointClientLink).filter(EndpointClientLink.client_id == cid).delete()
        db.delete(client)
        db.commit()
        db.close()
        return RedirectResponse(f"/clients", status_code=303)
    
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

    # Redirect the user to the resources page if none exist
    if db.query(Resource).count() == 0:
        db.close()
        return RedirectResponse(url="/resources", status_code=303)

    all_resources_query = db.query(Resource).order_by(Resource.name).all()
    all_resources_json_safe = [{"id": r.id, "name": r.name} for r in all_resources_query]
    # Fetch existing clients so the user can select or create clients when creating endpoint
    all_clients_query = db.query(Client).order_by(Client.name).all()
    all_clients_json_safe = [{"id": c.id, "name": c.name, "token": c.token} for c in all_clients_query]

    context = {
        "request": request,
        "all_resources": all_resources_json_safe,
        # default: include all resources for a new endpoint
        "associated_resources": all_resources_json_safe,
        "endpoint": None,
        "edit_action": None,
        "all_clients": all_clients_json_safe,
        # For a new endpoint there are no associated clients yet
        "associated_client_ids": []
    }
    db.close()
    return templates.TemplateResponse("endpoint_form.html", context)

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

    # Unique validation for endpoint path
    existing_ep = db.query(Endpoint).filter(Endpoint.path == path).first()
    if existing_ep:
        # prepare context similar to GET new endpoint form
        all_resources_query = db.query(Resource).order_by(Resource.name).all()
        all_resources_json_safe = [{"id": r.id, "name": r.name} for r in all_resources_query]
        all_clients_query = db.query(Client).order_by(Client.name).all()
        all_clients_json_safe = [{"id": c.id, "name": c.name, "token": c.token} for c in all_clients_query]
        context = {
            "request": request,
            "all_resources": all_resources_json_safe,
            "all_clients": all_clients_json_safe,
            "associated_resources": all_resources_json_safe,
            "endpoint": None,
            "edit_action": None,
            "associated_client_ids": [],
            "error": f"An endpoint with path '{path}' already exists."
        }
        db.close()
        return templates.TemplateResponse("endpoint_form.html", context)

    # Server-side: always ensure 'ref' and 'scope' exist in stored required_params for new endpoints
    final_required = sorted(list(set(['ref', 'scope'] + required_params)))

    new_endpoint = Endpoint(
        path=path,
        description=form_data.get("desc"),
        required_params=final_required,
        dev_hold_tasks = True if form_data.get('dev_hold_tasks') else False
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

    # --- Persist client associations (many-to-many) ---
    client_ids = form_data.getlist('client_ids') if hasattr(form_data, 'getlist') else []
    for cid in client_ids:
        try:
            cid_int = int(cid)
        except Exception:
            continue
        link = EndpointClientLink(endpoint_id=new_endpoint.id, client_id=cid_int)
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
    new_state = False
    if endpoint:
        endpoint.is_development_mode = not endpoint.is_development_mode
        new_state = bool(endpoint.is_development_mode)
        db.commit()
    db.close()
    # If dev mode was just turned ON, redirect to the rules page so the user
    # can immediately configure development rules for this endpoint.
    if new_state:
        return RedirectResponse(f"/endpoints/{eid}/rules", status_code=303)

    # Otherwise, return to the endpoints list (toggle was turned OFF)
    return RedirectResponse("/endpoints", status_code=303)
    
@app.post("/endpoints/{eid}/rules", include_in_schema=False, dependencies=[Depends(verify_session)])
async def rules_save(eid: int, 
    request: Request,
    is_new: bool = Form(False)
    ):
    db = SessionLocal()
    try:
        form_data = await request.form()

        # --- Persist endpoint-level development flags ---
        endpoint = db.query(Endpoint).get(eid)
        if not endpoint:
            db.close()
            return RedirectResponse("/endpoints", status_code=303)

        # Hidden input 'is_development_mode' is provided by the rules form (client-side toggle)
        is_dev_val = form_data.get('is_development_mode')
        endpoint.is_development_mode = True if is_dev_val in ['true', 'on', '1', 'True'] else False

        # dev_hold_tasks and dev_rules_enabled are sent as checkboxes
        endpoint.dev_hold_tasks = True if form_data.get('dev_hold_tasks') in ['true', 'on', '1', 'True'] else False
        endpoint.dev_rules_enabled = True if form_data.get('dev_rules_enabled') in ['true', 'on', '1', 'True'] else False

        # Server-side validation: Development Mode requires at least one of
        # dev_hold_tasks or dev_rules_enabled to be enabled. If not, reject
        # the change and redirect back with an error message.
        if endpoint.is_development_mode and not (endpoint.dev_hold_tasks or endpoint.dev_rules_enabled):
            db.close()
            # Preserve `is_new` query parameter when redirecting if present
            is_new_q = 'is_new=true' if form_data.get('is_new') else ''
            q = f"?{is_new_q}&error=dev_requires_hold_or_rules" if is_new_q else "?error=dev_requires_hold_or_rules"
            return RedirectResponse(f"/endpoints/{eid}/rules{q}", status_code=303)

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

        # Persist endpoint and rules
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

@app.get("/documentation", response_class=HTMLResponse, include_in_schema=False)
def documentation_page(dep=Depends(get_admin_or_client), request: Request = None):
    db = SessionLocal()
    try:
        # If client, render the standard documentation.html but instruct the
        # embedded Swagger UI to load a client-scoped OpenAPI JSON.
        doc_url = "/docs"
        is_client = dep.get('client') is not None
        context = {"client":dep.get('client'), "request": request, "doc_url": doc_url, "is_client": is_client}
        return templates.TemplateResponse("documentation.html", context)

    finally:
        db.close()

# -------------------------
# Client web session routes
# -------------------------
@app.get('/client/login', include_in_schema=False)
def client_login_form(request: Request):
    # Legacy client login URL now redirects to the unified login page.
    return RedirectResponse(url='/login', status_code=303)


@app.post('/client/login', include_in_schema=False)
async def client_login_submit(request: Request, token: str = Form(...)):
    db = SessionLocal()
    try:
        client = db.query(Client).filter(Client.token == token).first()
        if not client:
            # Render the unified login page, selecting the Client tab and showing
            # an inline error message so the user stays on the same form.
            return templates.TemplateResponse('login.html', {"request": request, "client_error": "Invalid token"}, status_code=401)
        # Set a cookie for client session and redirect to the unified dashboard '/'
        resp = RedirectResponse(url='/', status_code=303)
        resp.set_cookie(key=CLIENT_SESSION_COOKIE, value=client.token, httponly=True, path='/', samesite='lax')
        try:
            print(f"client_login_submit: set cookie {CLIENT_SESSION_COOKIE}={client.token!r}")
        except Exception:
            pass
        return resp
    finally:
        db.close()


@app.post('/client/logout', include_in_schema=False)
def client_logout():
    resp = RedirectResponse(url='/client/login', status_code=303)
    resp.delete_cookie(key=CLIENT_SESSION_COOKIE)
    return resp


@app.get('/client/dashboard', include_in_schema=False)
def client_dashboard_redirect(dep=Depends(verify_client_session)):
    """Compatibility route: redirect legacy /client/dashboard to the unified dashboard '/'.
    The client session is still validated by the dependency; after that the client
    should be able to access the root dashboard which accepts client sessions.
    """
    return RedirectResponse(url='/', status_code=303)


@app.get('/client/tasks', response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_client_session)])
def client_tasks(request: Request, endpoint_id: Optional[int] = None, client_token: Optional[str] = Cookie(None, alias=CLIENT_SESSION_COOKIE)):
    db = SessionLocal()
    try:
        client = db.query(Client).filter(Client.token == client_token).first()
        query = db.query(Task).filter(Task.client_id == client.id)
        if endpoint_id:
            query = query.filter(Task.endpoint_id == endpoint_id)
        tasks = query.order_by(Task.created_at.desc()).all()
        endpoints = [e for e in db.query(Endpoint).all()]
        return templates.TemplateResponse('client_tasks.html', {"request": request, "tasks": tasks, "endpoints": endpoints, "client": client})
    finally:
        db.close()


@app.post('/client/task/{tid}/retry', include_in_schema=False, dependencies=[Depends(verify_client_session)])
def client_task_retry(tid: int, client_token: Optional[str] = Cookie(None, alias=CLIENT_SESSION_COOKIE)):
    db = SessionLocal()
    try:
        client = db.query(Client).filter(Client.token == client_token).first()
        t = db.query(Task).get(tid)
        if not t or t.client_id != client.id:
            return RedirectResponse('/client/tasks', status_code=303)
        # reset task to pending and decrease retry_count
        t.status = 'pending'
        t.retry_count = 0
        db.commit()
        return RedirectResponse(f'/client/task/{tid}', status_code=303)
    finally:
        db.close()


# Add this new function with your other UI routes
@app.get("/tasks", response_class=HTMLResponse, include_in_schema=False)
def tasks_list(request: Request, dep=Depends(get_admin_or_client), status: Optional[str] = None, search: Optional[str] = None):
    db = SessionLocal()
    try:
        query = db.query(Task)

        # If client, restrict tasks to their own
        if dep.get('role') == 'client':
            client = dep.get('client')
            query = query.filter(Task.client_id == client.id)

        if status and status in ["pending", "success", "failed", "processing"]:
            query = query.filter(Task.status == status)
        
        if search:
            # Search within the source_payload JSON object
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    cast(Task.source_payload, String).ilike(search_term),
                    # Also search transformed/target payload, resource, client and endpoint path
                    cast(Task.target_payload, String).ilike(search_term),
                    Task.resource_name.ilike(search_term),
                    Task.client_name.ilike(search_term),
                    Task.endpoint_path.ilike(search_term),
                    Task.ref_id.ilike(search_term),
                    Task.scope.ilike(search_term)
                )
            )
        tasks = query.order_by(Task.created_at.desc()).all()
        # If the caller is a client, propagate `is_client` to the template so
        # the base navigation can hide admin-only links consistently.
        is_client = True if isinstance(dep, dict) and dep.get('role') == 'client' else False
        client = dep.get('client') if is_client else None
        context = {"request": request, "tasks": tasks, "current_status": status, "current_search": search, "is_client": is_client, "client": client}
        # Use TemplateResponse to ensure `is_client` is available to base.html
        return templates.TemplateResponse("tasks.html", context)
    finally:
        db.close()


# Batch delete tasks (top-level route)
@app.post("/tasks/batch_delete", include_in_schema=False, dependencies=[Depends(get_admin_or_client)])
async def tasks_batch_delete(request: Request):
    form = await request.form()
    # form.getlist is available on Starlette's FormData
    task_ids = form.getlist("task_ids") if hasattr(form, 'getlist') else [v for k, v in form.items() if k == 'task_ids']
    if not task_ids:
        return RedirectResponse("/tasks", status_code=303)
    db = SessionLocal()
    try:
        for tid in task_ids:
            try:
                t = db.query(Task).get(int(tid))
            except Exception:
                t = None
            if t:
                db.delete(t)
        db.commit()
    finally:
        db.close()
    return RedirectResponse("/tasks", status_code=303)
    
# --- Resources ---
@app.get("/resources", response_class=HTMLResponse,include_in_schema=False, dependencies=[Depends(verify_session)])
def resources_view():
    db = SessionLocal()
    resources = db.query(Resource).order_by(Resource.created_at).all()
    db.close()
    return render("resources.html", resources=resources)


@app.get("/resources/new", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(verify_session)])
def resource_new_form(request: Request):
    """Render the 'Create New Resource' form.

    The POST handler for creating a resource already exists at /resources/new.
    This GET handler simply renders the form template with sensible defaults so
    a browser can fetch the form via GET instead of receiving 405 Method Not Allowed.
    """
    context = {
        "request": request,
        "resource": None,
        "edit_action": "/resources/new",
        # Defaults used by the template when creating a new resource
        "default_subject": DEFAULT_FAILURE_EMAIL_SUBJECT,
        "default_template": DEFAULT_FAILURE_EMAIL_TEMPLATE
    }
    return templates.TemplateResponse("resource_form.html", context)

@app.post("/resources/edit/{rid}", include_in_schema=False, dependencies=[Depends(verify_session)])
def resource_edit_submit(
    request: Request,
    rid: int, 
    name: str = Form(...), 
    endpoint: str = Form(...), 
    header_keys: List[str] = Form([]),
    header_values: List[str] = Form([]),
    required_params: List[str] = Form([]),
    param_test_values: List[str] = Form([]),
    is_health_check_enabled: bool = Form(False),
    health_check_method: str = Form('simple'),
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
    # Unique name validation (allow same name if editing the same resource)
    existing = db.query(Resource).filter(Resource.name == name, Resource.id != rid).first()
    if existing:
        # re-fetch resource for context
        resource = db.query(Resource).get(rid)
        context = {"request": request, "resource": resource, "error": f"Resource with name '{name}' already exists.", "default_subject": DEFAULT_FAILURE_EMAIL_SUBJECT, "default_template": DEFAULT_FAILURE_EMAIL_TEMPLATE.strip(), "edit_action": f"/resources/edit/{rid}"}
        db.close()
        return templates.TemplateResponse('resource_form.html', context)

    r.name = name
    r.endpoint = endpoint
    r.headers = {key: value for key, value in zip(header_keys, header_values) if key} or None
    r.required_params = required_params or None
    r.is_health_check_enabled = is_health_check_enabled
    r.health_check_test_values = {key: value for key, value in zip(required_params, param_test_values) if key and value} or None
    # Persist selected health check method
    r.health_check_method = health_check_method or r.health_check_method
    
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

    # Server-side validation for edit: require test values when HC is enabled and method is 'actual'
    if r.is_health_check_enabled and r.health_check_method == 'actual':
        rp = required_params or []
        tv = r.health_check_test_values or {}
        missing = [p for p in rp if p and not tv.get(p)]
        if missing:
            db.close()
            raise HTTPException(status_code=400, detail=f"Health check method 'actual' requires test values for parameters: {', '.join(missing)}")
    
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

@app.post("/resources/duplicate/{rid}", include_in_schema=False, dependencies=[Depends(verify_session)])
def resources_duplicate(rid: int):
    """
    Duplicate a resource (copy all fields except health-check related data).
    The new resource will be created with is_active=False and a unique name.
    """
    db = SessionLocal()
    src = db.query(Resource).get(rid)
    if not src:
        db.close()
        return RedirectResponse(url="/resources", status_code=303)

    # Base name for the copy
    base_name = f"{src.name} copy"
    new_name = base_name
    suffix = 1
    # Ensure uniqueness
    while db.query(Resource).filter(Resource.name == new_name).first():
        suffix += 1
        new_name = f"{base_name} {suffix}"

    # Create a new Resource object copying fields except health-check specific ones
    # NOTE: We intentionally DO NOT copy any health-check history/logs.
    # Health check logs are stored in the HealthCheckLog table and should
    # not be duplicated when cloning a resource.
    new_res = Resource(
        name=new_name,
        endpoint=src.endpoint,
        headers=src.headers,
        required_params=src.required_params,
        is_active=False,
        healthy=False,
        is_health_check_enabled=False,
        health_check_test_values=None,
        health_check_method='simple',
        health_check_interval_min=src.health_check_interval_min or 60,
        health_check_interval_max=src.health_check_interval_max or 90,
        notification_email=src.notification_email,
        notification_subject=src.notification_subject,
        notification_template=src.notification_template,
        send_failure_email=src.send_failure_email,
        send_failure_webhook=src.send_failure_webhook,
        failure_webhook_url=src.failure_webhook_url,
        failure_webhook_headers=src.failure_webhook_headers,
        failure_webhook_payload=src.failure_webhook_payload
    )
    db.add(new_res)
    db.commit()
    db.close()
    return RedirectResponse(url="/resources", status_code=303)


@app.post("/resources/new", include_in_schema=False, dependencies=[Depends(verify_session)])
def resources_new_submit(
    request: Request,
    name: str = Form(...),
    endpoint: str = Form(...),
    header_keys: List[str] = Form([]),
    header_values: List[str] = Form([]),
    required_params: List[str] = Form([]),
    param_test_values: List[str] = Form([]),
    is_health_check_enabled: bool = Form(False),
    health_check_method: str = Form('simple'),
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
    multiplier = {'seconds': 1, 'minutes': 60, 'hours': 3600}.get(interval_unit, 60)
    min_seconds = interval_min * multiplier
    max_seconds = interval_max * multiplier
        
    headers_dict = {key: value for key, value in zip(header_keys, header_values) if key}
    test_values_dict = {key: value for key, value in zip(required_params, param_test_values) if key and value}

    # Server-side validation: if health check is enabled and method is 'actual', require test values for all required params
    if is_health_check_enabled and health_check_method == 'actual':
        missing = [p for p in (required_params or []) if p and not test_values_dict.get(p)]
        if missing:
            db.close()
            raise HTTPException(status_code=400, detail=f"Health check method 'actual' requires test values for parameters: {', '.join(missing)}")

    # Quick endpoint validity check: choose method based on requested health_check_method
    healthy = False
    last_checked = None
    next_health = None
    try:
        if health_check_method == 'actual' and test_values_dict:
            temp_res = Resource(
                name=name,
                endpoint=endpoint,
                headers=headers_dict or None,
                health_check_test_values=test_values_dict or None
            )
            healthy, status_code, resp_text = perform_health_check(temp_res)
        else:
            resp = requests.get(endpoint, headers=headers_dict or None, timeout=5)
            healthy = 200 <= resp.status_code < 300
    except Exception:
        healthy = False
    last_checked = datetime.utcnow()
    if is_health_check_enabled:
        next_health = datetime.utcnow()

    r = Resource(
        name=name,
        endpoint=endpoint,
        headers=headers_dict or None,
        required_params=required_params or None,
        is_active=True,
        healthy=healthy,
        last_checked=last_checked,
        is_health_check_enabled=is_health_check_enabled,
        health_check_test_values=test_values_dict or None,
        health_check_method=health_check_method,
        health_check_interval_min=min_seconds,
        health_check_interval_max=max_seconds,
        next_health_check_at=next_health,
        notification_email=notification_email,
        notification_subject=notification_subject or DEFAULT_FAILURE_EMAIL_SUBJECT,
    notification_template=notification_template or DEFAULT_FAILURE_EMAIL_TEMPLATE,
        # --- ADDED LOGIC ---
        send_failure_email=send_failure_email,
        send_failure_webhook=send_failure_webhook,
        failure_webhook_url=failure_webhook_url,
        failure_webhook_headers=json.loads(failure_webhook_headers) if failure_webhook_headers and failure_webhook_headers.strip() else None,
        failure_webhook_payload=json.loads(failure_webhook_payload) if failure_webhook_payload and failure_webhook_payload.strip() else None
    )
    # Unique resource name validation
    existing_res = db.query(Resource).filter(Resource.name == name).first()
    if existing_res:
        db.close()
        context = {
            "request": request,
            "resource": None,
            "error": f"Resource with name '{name}' already exists.",
            "default_subject": DEFAULT_FAILURE_EMAIL_SUBJECT,
            "default_template": DEFAULT_FAILURE_EMAIL_TEMPLATE.strip(),
            "edit_action": "/resources/new"
        }
        return templates.TemplateResponse('resource_form.html', context)

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
    # Provide the action URL for the edit form
    context["edit_action"] = f"/resources/edit/{resource.id}"
    return templates.TemplateResponse("resource_form.html", context)

@app.get("/resources/{resource_id}/details", include_in_schema=False, dependencies=[Depends(verify_session)])
def get_resource_details(resource_id: int):
    db = SessionLocal()
    resource = db.query(Resource).get(resource_id)
    db.close()
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")
    return {"id": resource.id, "name": resource.name, "required_params": resource.required_params or []}

@app.get("/resources/{rid}/history", response_class=HTMLResponse, include_in_schema=False, dependencies=[Depends(get_admin_or_client)])
def resource_history(rid: int, dep=Depends(get_admin_or_client)):
    db = SessionLocal()
    resource = db.query(Resource).get(rid)
    if not resource: 
        raise HTTPException(status_code=404, detail="Resource not found")
    
    # Fetches the last 5 logs in descending order
    logs = db.query(HealthCheckLog).filter(HealthCheckLog.resource_id == rid).order_by(HealthCheckLog.created_at.desc()).limit(5).all()
    is_client = dep.get('role') == 'client'
    client_obj = None
    if isinstance(dep, dict) and is_client:
        client_obj = dep.get('client')
    db.close()
    return render("health_history.html", resource=resource,client=client_obj, is_client=is_client, logs=logs)

@app.post("/resources/{rid}/check", include_in_schema=False, dependencies=[Depends(verify_session)])
def manual_health_check(rid: int):
    request_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        resource = db.query(Resource).get(rid)
        if not resource:
            raise HTTPException(status_code=404, detail="Resource not found")

        print(f"manual_check start: rid={rid} request_id={request_id}")

        # In-memory short-lived guard: if a manual check for this resource started recently, reject
        now = time.time()
        guard = manual_check_locks.get(rid)
        if guard and now - guard < 2.0:
            # Return both 'detail' and 'message' for compatibility with client JS
            return JSONResponse({
                'success': None,
                'detail': 'check already running',
                'message': 'check already running',
                'request_id': request_id
            }, status_code=409)

        # Attempt to acquire per-resource file lock non-blocking; if not acquired, another check is running
        lock_fh = _acquire_resource_lock(rid)
        if not lock_fh:
            # can't get lock
            manual_check_locks[rid] = now
            print(f"manual_check lock failed: rid={rid} request_id={request_id}")
            # Return both 'detail' and 'message' so client code can display a helpful message
            return JSONResponse({
                'success': None,
                'detail': 'check already running (lock)',
                'message': 'check already running (lock)',
                'request_id': request_id
            }, status_code=409)

        # mark in-memory guard while we run
        manual_check_locks[rid] = now
        try:
            # Respect the resource's configured health_check_method for manual checks
            method_used = (resource.health_check_method or 'simple')
            if method_used == 'simple':
                # First, use the lightweight non-invasive probe (TCP + OPTIONS/HEAD)
                try:
                    is_healthy, status_code, response_text = perform_health_check(resource)
                except Exception as e:
                    is_healthy, status_code, response_text = False, None, str(e)

                # If the user explicitly wants to confirm by making a request, do a minimal POST probe
                # This will not be used to mark the resource healthy unless it returns 2xx.
                try:
                    probe_headers = (resource.headers.copy() if resource.headers else {})
                    # Ensure a content-type for the probe
                    if not any(k.lower() == 'content-type' for k in probe_headers.keys()):
                        probe_headers['Content-Type'] = 'application/json'

                    probe_payload = resource.health_check_test_values if resource.health_check_test_values is not None else {}
                    # Ensure probe_payload is a dict; fall back to {}
                    if not isinstance(probe_payload, dict):
                        probe_payload = {}

                    # Create a sanitized dummy payload: replace potentially side-effecting values
                    def _sanitize_value(v):
                        if isinstance(v, str):
                            # Replace likely phone numbers with a neutral numeric placeholder
                            if re.match(r"^\+?\d{6,}$", v):
                                return "0000000000"
                            # Avoid returning long text or service-specific samples — use a short generic placeholder
                            if len(v) > 200:
                                return "dummy_string"
                            return "dummy_string"
                        elif isinstance(v, (int, float)):
                            return 0
                        elif isinstance(v, dict):
                            return {k: _sanitize_value(val) for k, val in v.items()}
                        elif isinstance(v, list):
                            return [_sanitize_value(x) for x in v]
                        else:
                            return str(v)

                    sanitized_payload = {k: _sanitize_value(v) for k, v in probe_payload.items()} if probe_payload else {}

                    # Add marker header to indicate this is a dummy health-check probe
                    probe_headers['X-Health-Check'] = 'dummy'

                    resp_probe = requests.post(resource.endpoint, headers=probe_headers, json=sanitized_payload, timeout=8)
                    probe_code = getattr(resp_probe, 'status_code', None)
                    probe_text = getattr(resp_probe, 'text', '')

                    # Attach probe result to response_text for visibility
                    response_text = (response_text or '') + f"\nPOST-probe:{probe_code}:{probe_text[:1000]}"
                    # Only mark healthy if probe returned 2xx (do not auto-promote on 3xx/4xx/5xx)
                    if 200 <= probe_code < 300:
                        is_healthy = True
                        status_code = probe_code
                    else:
                        # prefer to show the actual probe status code if available
                        status_code = probe_code or status_code
                except Exception as e:
                    # If probe failed, append the error but keep prior probe results
                    response_text = (response_text or '') + f"\nPOST-probe-error:{str(e)}"
            else:
                is_healthy, status_code, response_text = perform_health_check(resource)

            # Create and save the log entry first
            new_log = HealthCheckLog(
                resource_id=resource.id, is_success=is_healthy,
                status_code=status_code, response_text=response_text,
                method=method_used
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
            try:
                print(f"manual_check saved log: rid={rid} request_id={request_id} log_id={new_log.id}")
            except Exception:
                print(f"manual_check saved log (id unknown): rid={rid} request_id={request_id}")

            # --- Update Resource summary explicitly ---
            db.query(Resource).filter(Resource.id == rid).update({
                "healthy": is_healthy,
                "last_checked": datetime.utcnow(),
                "latest_health_check_result": { "success": is_healthy, "status_code": status_code, "detail": response_text[:200] },
                "next_health_check_at": datetime.utcnow() + timedelta(seconds=random.randint(resource.health_check_interval_min, resource.health_check_interval_max))
            })
            db.commit()

            # Prune old logs
            logs_to_keep_ids = [log_id for log_id, in db.query(HealthCheckLog.id).filter(HealthCheckLog.resource_id == rid).order_by(HealthCheckLog.created_at.desc()).limit(5).all()]
            if logs_to_keep_ids:
                db.query(HealthCheckLog).filter(HealthCheckLog.resource_id == rid, ~HealthCheckLog.id.in_(logs_to_keep_ids)).delete(synchronize_session=False)
                db.commit()

            return JSONResponse({ "success": is_healthy, "status_code": status_code, "detail": response_text[:200], 'request_id': request_id })
        finally:
            # release file lock
            try:
                lock_fh.close()
            except Exception:
                pass
            # clear in-memory guard after a short delay
            def _clear_guard(rid_local):
                try:
                    time.sleep(1.0)
                except Exception:
                    pass
                manual_check_locks.pop(rid_local, None)

            t = Thread(target=_clear_guard, args=(rid,))
            t.daemon = True
            t.start()
    finally:
        db.close()


# --- Endpoints ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard(request: Request, dep=Depends(get_admin_or_client)):
    db = SessionLocal()
    try:
        # Admin view: full data
        if dep.get('role') == 'admin':
            # 1. Fetch only resources that are marked as active
            active_resources = db.query(Resource).filter(Resource.is_active == True).order_by(Resource.name).all()

            # 2. Fetch all active endpoints and determine their status
            active_endpoints_query = db.query(Endpoint).filter(Endpoint.is_active == True).order_by(Endpoint.path).all()
            active_endpoints_data = []
            for e in active_endpoints_query:
                active_endpoints_data.append({
                    "path": e.path,
                    "status": get_endpoint_status(db, e),
                    # indicate development mode so templates can render a badge
                    "is_dev": bool(getattr(e, 'is_development_mode', False))
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
                "endpoints": active_endpoints_data,
                "tasks": tasks,
                "stats": stats
            }
            return templates.TemplateResponse("dashboard.html", context)

        # Client view: scoped to client-associated endpoints/resources/tasks
        client = dep.get('client')
        if not client:
            # fallback redirect to client login
            raise HTTPException(status_code=307, headers={"Location": "/client/login"})

        # find associated endpoints via EndpointClientLink
        links = db.query(EndpointClientLink).filter(EndpointClientLink.client_id == client.id).all()
        endpoint_ids = [l.endpoint_id for l in links]
        endpoints_q = db.query(Endpoint).filter(Endpoint.id.in_(endpoint_ids)).all() if endpoint_ids else []

        # Build endpoints UI data similar to admin homepage (path, status, is_dev)
        endpoints_ui = []
        for e in endpoints_q:
            endpoints_ui.append({
                "path": e.path,
                "status": get_endpoint_status(db, e),
                "is_dev": bool(getattr(e, 'is_development_mode', False))
            })

        # Resources: only those linked to the client's endpoints
        resource_links = db.query(EndpointResourceLink).filter(EndpointResourceLink.endpoint_id.in_(endpoint_ids)).all() if endpoint_ids else []
        resource_ids = list(dict.fromkeys([rl.resource_id for rl in resource_links])) if resource_links else []
        resources = db.query(Resource).filter(Resource.id.in_(resource_ids), Resource.is_active == True).order_by(Resource.name).all() if resource_ids else []

        # tasks stats and recent tasks for this client
        tasks = db.query(Task).filter(Task.client_id == client.id).order_by(Task.created_at.desc()).limit(10).all()
        stats = {
            "total": db.query(Task).filter(Task.client_id == client.id).count(),
            "pending": db.query(Task).filter(Task.client_id == client.id, Task.status == "pending").count(),
            "success": db.query(Task).filter(Task.client_id == client.id, Task.status == "success").count(),
            "failed": db.query(Task).filter(Task.client_id == client.id, Task.status == "failed").count()
        }

        context = {"request": request, "client": client, "resources": resources, "endpoints": endpoints_ui, "tasks": tasks, "stats": stats, "is_client": True}
        return templates.TemplateResponse("dashboard.html", context)
    finally:
        db.close()
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
        "all_resources": all_resources_json_safe,
        "edit_action": f"/endpoints/edit/{endpoint.id}",
        "all_clients": [{"id": c.id, "name": c.name, "token": c.token} for c in db.query(Client).order_by(Client.name).all()],
        "associated_client_ids": [l.client_id for l in db.query(EndpointClientLink).filter(EndpointClientLink.endpoint_id == eid).all()]
    }
    db.close()
    return templates.TemplateResponse("endpoint_form.html", context)

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
    # For edit: persist exactly what user provided (allow removing ref/scope)
    endpoint.required_params = required_params or []

    # Rebuild Resource Links based on the new sequence
    db.query(EndpointResourceLink).filter(EndpointResourceLink.endpoint_id == eid).delete()
    for i, resource_id in enumerate(resource_sequence):
        if resource_id: # Ensure we don't save empty selections
            link = EndpointResourceLink(endpoint_id=eid, resource_id=resource_id, sequence=i)
            db.add(link)
    
    # --- Update client associations: only update if the form included client selection ---
    # If the edit form did not include 'client_ids' (for example when only
    # reordering resources), preserve the existing client links.
    if 'client_ids' in form_data:
        db.query(EndpointClientLink).filter(EndpointClientLink.endpoint_id == eid).delete()
        client_ids = form_data.getlist('client_ids') if hasattr(form_data, 'getlist') else []
        for cid in client_ids:
            try:
                cid_int = int(cid)
            except Exception:
                continue
            db.add(EndpointClientLink(endpoint_id=eid, client_id=cid_int))

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

            # Delete all association links between this endpoint and clients
            db.query(EndpointClientLink).filter(EndpointClientLink.endpoint_id == eid).delete(synchronize_session=False)

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


@app.post('/endpoints/duplicate/{eid}', include_in_schema=False, dependencies=[Depends(verify_session)])
def endpoints_duplicate(eid: int):
    """
    Duplicate an endpoint including its resource links, parameter rules,
    development rules, and client associations. The new endpoint will have
    " copy" appended to its path to help ensure uniqueness; if that path
    exists, a numeric suffix is appended.
    """
    db = SessionLocal()
    new_id = None
    try:
        src = db.query(Endpoint).get(eid)
        if not src:
            db.close()
            return RedirectResponse('/endpoints', status_code=303)

        # Build a unique path for the duplicate
        base_path = f"{src.path} copy"
        new_path = base_path
        suffix = 1
        while db.query(Endpoint).filter(Endpoint.path == new_path).first():
            suffix += 1
            new_path = f"{base_path} {suffix}"

        new_ep = Endpoint(
            path=new_path,
            description=src.description,
            required_params=src.required_params,
            is_development_mode=src.is_development_mode,
            dev_hold_tasks=src.dev_hold_tasks,
            dev_rules_enabled=src.dev_rules_enabled,
            is_active=src.is_active,
            dev_values=src.dev_values
        )
        db.add(new_ep)
        db.commit()
        db.refresh(new_ep)

        # Duplicate resource links with same sequence
        links = db.query(EndpointResourceLink).filter(EndpointResourceLink.endpoint_id == eid).order_by(EndpointResourceLink.sequence).all()
        for link in links:
            db.add(EndpointResourceLink(endpoint_id=new_ep.id, resource_id=link.resource_id, sequence=link.sequence))

        # Duplicate parameter rules
        param_rules = db.query(ParameterRule).filter(ParameterRule.endpoint_id == eid).all()
        for pr in param_rules:
            db.add(ParameterRule(endpoint_id=new_ep.id, resource_id=pr.resource_id, target_key=pr.target_key, source_type=pr.source_type, source_value=pr.source_value))

        # Duplicate development rules
        dev_rules = db.query(DevelopmentRule).filter(DevelopmentRule.endpoint_id == eid).all()
        for dr in dev_rules:
            db.add(DevelopmentRule(endpoint_id=new_ep.id, condition_param=dr.condition_param, condition_value=dr.condition_value, target_param=dr.target_param, override_value=dr.override_value))

        # Duplicate client links
        client_links = db.query(EndpointClientLink).filter(EndpointClientLink.endpoint_id == eid).all()
        for cl in client_links:
            db.add(EndpointClientLink(endpoint_id=new_ep.id, client_id=cl.client_id))

        db.commit()
        # capture the new endpoint id before closing the session to avoid DetachedInstanceError
        new_id = new_ep.id
    finally:
        db.close()

    if new_id:
        return RedirectResponse(f"/endpoints/edit/{new_id}", status_code=303)
    return RedirectResponse('/endpoints', status_code=303)

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
@app.get("/task/{tid}", response_class=HTMLResponse, include_in_schema=False)
def task_detail(tid: int, dep=Depends(get_admin_or_client)):
    db = SessionLocal()
    try:
        task = db.query(Task).get(tid)
        if not task:
            raise HTTPException(404)

        # If client, ensure they own the task
        if dep.get('role') == 'client':
            client = dep.get('client')
            if task.client_id != client.id:
                # Redirect clients to client login (or tasks list)
                raise HTTPException(status_code=307, headers={"Location": "/client/login"})

        logs = db.query(Log).filter(Log.task_id == tid).order_by(Log.created_at.asc()).all()
        # NEW: Fetch the attempt logs
        is_client = dep.get('role') == 'client'
        attempt_logs = db.query(TaskAttemptLog).filter(TaskAttemptLog.task_id == tid).order_by(TaskAttemptLog.attempt_at.asc()).all()
        # Ensure 'client' is always present in context for base.html
        client_obj = None
        if isinstance(dep, dict) and dep.get('role') == 'client':
            client_obj = dep.get('client')
        context = dict(task=task, logs=logs, attempt_logs=attempt_logs, is_client=is_client, client=client_obj)
        return render("task_detail.html", **context)
    finally:
        db.close()


@app.post("/task/{tid}/retry", include_in_schema=False, dependencies=[Depends(get_admin_or_client)])
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

@app.post("/task/{tid}/stop", include_in_schema=False, dependencies=[Depends(get_admin_or_client)])
def stop_task(tid: int, dep=Depends(get_admin_or_client)):
    db = SessionLocal()
    try:
        task = db.query(Task).get(tid)
        if not task:
            db.close()
            return JSONResponse({"detail": "Task not found."}, status_code=404)
        # Only allow stop if pending
        if task.status != "pending":
            db.close()
            return JSONResponse({"detail": "Only pending tasks can be stopped."}, status_code=400)
        task.status = "stopped"
        task.last_error = "Stopped by user."
        db.commit()
        db.close()
        return JSONResponse({"success": True, "message": "Task stopped."})
    except Exception as e:
        db.close()
        return JSONResponse({"detail": str(e)}, status_code=500)

@app.post("/task/{tid}/delete", include_in_schema=False, dependencies=[Depends(get_admin_or_client)])
def delete_task(tid: int, dep=Depends(get_admin_or_client)):
    db = SessionLocal()
    try:
        task = db.query(Task).get(tid)
        if not task:
            db.close()
            return JSONResponse({"detail": "Task not found."}, status_code=404)
        # Only allow delete if failed or success
        if task.status in ["processing"]:
            db.close()
            return JSONResponse({"detail": "Processing tasks can not be deleted."}, status_code=400)
        # Delete related logs
        db.query(TaskLog).filter(TaskLog.task_id == tid).delete()
        db.query(TaskAttemptLog).filter(TaskAttemptLog.task_id == tid).delete()
        db.delete(task)
        db.commit()
        db.close()
        return JSONResponse({"success": True, "message": "Task deleted."})
    except Exception as e:
        db.close()
        return JSONResponse({"detail": str(e)}, status_code=500)
    
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
        "task_delay_max": get_setting(db, "task_delay_max", "5"),
        # NEW: Task cleanup settings
        "enable_task_cleanup": get_setting(db, "enable_task_cleanup", "false"),
        "task_cleanup_days": get_setting(db, "task_cleanup_days", "30")
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
    task_delay_max: int = Form(5),
    # NEW: Task cleanup parameters
    enable_task_cleanup: bool = Form(False),
    task_cleanup_days: int = Form(30)
):
    db = SessionLocal()
    settings_data = {
        "smtp_server": smtp_server, "smtp_port": str(smtp_port),
        "smtp_username": smtp_username, "smtp_password": smtp_password,
        "email_from": email_from, "smtp_encryption": smtp_encryption,
        "max_failure_attempts": str(max_failure_attempts),
        "task_delay_min": str(task_delay_min),
        "task_delay_max": str(task_delay_max),
        # NEW: Add cleanup settings
        "enable_task_cleanup": "true" if enable_task_cleanup else "false",
        "task_cleanup_days": str(task_cleanup_days)
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
    request: Request = None,
    x_api_token: Optional[str] = Header(None, description="The authentication token for the client.")
):
    epath = epath.strip().lstrip("/")
    db = SessionLocal()
    try:
        endpoint = db.query(Endpoint).filter(Endpoint.path == epath).first()
        if not endpoint or not endpoint.is_active:
            raise HTTPException(status_code=404, detail="Endpoint not found")

        # Support flexible header names for the API token. Priority:
        # 1) Explicit x_api_token header (FastAPI Header parameter)
        # 2) Other common header names: Token, Access-Token, Access_Token, token
        # 3) Authorization: Bearer <token>
        token = x_api_token
        if not token and request is not None:
            header_candidates = [
                'token', 'access_token', 'access-token', 'access_token',
                'x-api-token', 'x_api_token', 'authorization'
            ]
            for hn in header_candidates:
                hv = request.headers.get(hn)
                if not hv:
                    # try case-insensitive fallback (headers are case-insensitive but ensure coverage)
                    hv = request.headers.get(hn.lower()) or request.headers.get(hn.upper())
                if hv:
                    if hn == 'authorization':
                        # support 'Bearer <token>' or plain token
                        parts = hv.split()
                        if len(parts) == 2 and parts[0].lower() == 'bearer':
                            token = parts[1]
                        else:
                            token = hv
                    else:
                        token = hv
                    break

        if not token:
            raise HTTPException(status_code=401, detail="API token is missing")

        # Validate token against clients linked to this endpoint via the association table
        client = db.query(Client).join(EndpointClientLink, Client.id == EndpointClientLink.client_id)\
            .filter(Client.token == token, EndpointClientLink.endpoint_id == endpoint.id).first()
        if not client:
            # Fallback for legacy single-column clients.endpoint_id for older DBs
            legacy_client = db.query(Client).filter(Client.token == token, Client.endpoint_id == endpoint.id).first()
            if not legacy_client:
                raise HTTPException(status_code=403, detail="Invalid API token for this endpoint")
            client = legacy_client

        ref_id = payload.get("ref")
        scope = payload.get("scope")
        if not ref_id or not scope:
            raise HTTPException(status_code=400, detail="Payload must include 'ref' and 'scope' fields")
        
        if endpoint.required_params:
            for required_key in endpoint.required_params:
                if required_key not in payload:
                    raise HTTPException(status_code=400, detail=f"Missing required parameter: {required_key}")
        
        # Get the global max_retries setting from the database
        max_retries = get_setting_int(db, "max_failure_attempts", 5)
        min_delay = get_setting_int(db, "task_delay_min", 1)
        max_delay = get_setting_int(db, "task_delay_max", 5)
        # Ensure we have a valid range for randint
        if max_delay < min_delay:
            max_delay = min_delay
        delay = random.randint(min_delay, max_delay)
        # Create the task in a 'pending' state by default. Client-level
        # rules are evaluated first; if any active client rule matches the
        # incoming payload the task will be marked as 'dev' (held). Next we
        # fall back to per-client global flag, then endpoint defaults.
        initial_status = "pending"
        # If no client rule matched, check global client-level dev hold flag, then endpoint

        # Only hold a task for a client when dev mode is explicitly enabled AND
        # the client has opted to 'hold' incoming tasks.
        if client.dev_hold_tasks or (endpoint.is_development_mode and endpoint.dev_hold_tasks):
            initial_status = 'dev'

        t = Task(
            endpoint_id=endpoint.id,
            endpoint_path=endpoint.path,
            source_payload=payload,
            client_name=client.name,
            client_id=client.id if client else None,
            ref_id=ref_id,
            scope=scope,
            status=initial_status,
            attempts=0,
            retry_count=0,
            max_retries=max_retries,
            delay=delay,
        )
        try:
            print(f"Final payload before task create: {json.dumps(payload) if isinstance(payload, dict) else str(payload)}")
        except Exception:
            pass
        db.add(t)
        db.commit()
        db.refresh(t)

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
