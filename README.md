# What is <img height="25" width="25" src="https://github.com/ariehariady/dispatchapy/blob/main/statics/favicon.png" alt="Dispatchapy Logo" width="80"> Dispatchapy?

**Dispatchapy** is a resilient, high-availability API dispatch gateway. It is designed to be a single, reliable entry point for sending notifications and webhooks through multiple providers. With features like automatic failover, health checks, per-client documentation, and a full UI for configuration, it ensures your critical communications are always delivered.

### Key Features

-   ✅ **High Availability:** Automatically fails over to healthy resources in a user-defined sequence.
-   🩺 **Health Checks:** Actively monitors the status of all resources and provides a detailed history.
-   📧 **Failure Notifications:** Sends email alerts or triggers webhooks when a resource goes down.
-   🔧 **Dynamic Configuration:** Manage Resources, Endpoints, Clients, Rules and Development Overrides through a clean web interface.
-   🔒 **Flexible Authentication:** Clients authenticate with tokens; the gateway accepts `X-API-Token` (recommended) as well as several alternative header names and `Authorization: Bearer <token>` for compatibility.
-   📚 **Per-User Documentation:** The `/documentation` page loads a client-scoped OpenAPI JSON so clients see only endpoints they are allowed to call, and the docs can auto-inject the client token for "Try it out".
-   🧪 **Development Mode & Dev Rules:** Hold incoming tasks for manual inspection or apply client/endpoint-level override rules to transform payloads for testing without changing client code.
-   🗂️ **Task Management & Retry:** Tasks progress through Pending → Processing → Success/Failed; workers support retrying and manual retry/stop/delete operations in the UI (with guards to prevent deleting processing tasks).
-   🔎 **Task Search:** The Tasks UI search can match fields in the original (source) payload, transformed (target) payload, resource name, client name, endpoint path, ref and scope.
-   🐳 **Containerized:** Easy to deploy with `docker-compose`.

---

## 🚀 Quick Start with Docker Compose

This is the recommended way to run Dispatchapy for both development and production.

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

### Running the Application

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ariehariady/dispatchapy.git
    cd dispatchapy
    ```

2.  **Configure the Password:**
    Open the `docker-compose.yml` file and set `GATEWAY_ADMIN_PASSWORD` to a strong, secret password.

3.  **Build and Run:**
    Run the following command from the project's root directory. This will build the Docker image and start the container in the background.
    ```bash
    docker-compose up -d --build
    ```

4.  **Access the Gateway:**
    The application is now running!
    - **UI:** Open your browser to `http://localhost:8000`

---

## 📖 Functional Documentation

Dispatchapy is configured entirely through the web interface. Below are the core concepts and current behavior implemented by the code and templates.

### 1. Resources
A **Resource** is a destination API endpoint that you want to send data to. Each resource supports:
- **Health Checks:** Periodic probes (simple or actual POST) mark a resource healthy or down; history is stored and visible in the UI.
- **Notifications:** Configure email and/or webhook notifications on failure.

### 2. Endpoints
An **Endpoint** is a custom API route on the gateway (eg. `/api/send_alert`). Endpoints include:
- **Associated Resources:** Ordered list used for failover; workers will pick the first healthy resource.
- **Required Incoming Parameters:** Define required keys for incoming JSON payloads.
- **Development Mode:** Toggle per-endpoint dev mode and attach development rules to override or conditionally transform incoming data.

### 3. Clients & Authentication
Calls to `/api/{endpoint}` must be authenticated.
- **Clients:** Create named clients and assign them to endpoints. Each client has a secret token.
- **Accepted Headers:** The gateway accepts `X-API-Token` (recommended) and also recognizes several alternative header names and `Authorization: Bearer <token>` for compatibility.
- **Client Sessions & Docs:** Clients can login in the UI (cookie-based session). The `/documentation` page serves a client-scoped OpenAPI JSON so clients only see their allowed `/api/*` paths; the embedded Swagger UI attempts to auto-inject the client token for Try-It-Out requests.

### 4. Rules & Development Overrides
The **Rules** UI lets you:
- **Parameter Mapping:** Map keys from the client's payload into the shape required by each resource (the worker constructs a transformed `target_payload`).
- **Development Rules (Dev Mode):** Define client-level and endpoint-level override rules that can transform or replace fields when Dev Mode is active or when client dev rules are enabled. There is also an option to hold incoming tasks for manual inspection (`dev_hold_tasks`).

### 5. Task Lifecycle & Management
- **States:** Pending → Processing → Success / Failed. Admin and client UIs provide controls to Stop (pending → stopped), Retry, and Delete tasks.
- **Delete Guards:** Deleting a processing task via the UI is prevented; batch delete in the tasks list will block deletion if any selected task is `processing`.
- **Logs & Attempts:** Each task stores attempt logs and a transformed `target_payload` so you can inspect both source and transformed payloads in the UI.

---

## ⚙️ How to Use: A Step-by-Step Guide

Follow these steps to configure an endpoint and a client.

### Step 1: Configure SMTP (Optional)
For failure notifications, configure your SMTP settings in **Settings**.

### Step 2: Create Resources
Go to **Resources → + New Resource** and define endpoint URL, optional headers, and required parameters. Save one or more resources (primary + backups).

### Step 3: Create an Endpoint
Go to **Endpoints → + New Endpoint**. Define the endpoint path (used as the `{epath}` under `/api/{epath}`), set required incoming parameters, and attach associated resources. Save and configure mapping rules.

### Step 4: Configure Rules
Use the **Rules** page to map incoming keys into the required resource keys, then save development rules if you want overrides for testing.

### Step 5: Create a Client
Go to **Clients → + New Client**, give it a name and copy the generated token. You can associate clients with endpoints so they only see/use those endpoints.

### Step 6: Call the Endpoint
POST JSON payload to `/api/<endpoint_path>`. Include your token in the `X-API-Token` header (or in an accepted alternative header). Example:

```bash
curl -X POST "http://localhost:8000/api/log_user_event" \
  -H "Content-Type: application/json" \
  -H "X-API-Token: YOUR_GENERATED_CLIENT_TOKEN" \
  -d '{
    "ref": "req-987654",
    "scope": "production",
    "event_id": "evt_abc123",
    "user_id": 42,
    "event_type": "USER_LOGIN_SUCCESS"
  }'
```

---

## � Tips & Notes
- The `/documentation` UI provides Swagger for exploring endpoints; logged-in clients see only their allowed `/api/*` paths and the UI will try to inject their client token for "Try it out".
- The Tasks list search is broad: it searches the source payload, the transformed target payload, resource and client names, endpoint path, ref and scope. For heavy usage consider adding indexes or a full-text approach.
- Development Mode is powerful — use `dev_hold_tasks` to capture incoming requests without processing, or dev rules to rewrite payloads for safe testing.
- Deleting processing tasks is blocked by the UI (and server-side guards). Admins can review task state and stop before deleting.

---

## LICENSE

Copyright (c) 2025 Arie Hariady

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.

