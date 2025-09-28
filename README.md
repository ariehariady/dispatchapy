# What is <img height="25" width="25" src="https://github.com/ariehariady/dispatchapy/blob/main/statics/favicon.png" alt="Dispatchapy Logo" width="80"> Dispatchapy?

**Dispatchapy** is a resilient, high-availability API dispatch gateway. It is designed to be a single, reliable entry point for sending notifications and webhooks through multiple providers. With features like automatic failover, health checks, and a full UI for configuration, it ensures your critical communications are always delivered.

### Key Features

-   ✅ **High Availability:** Automatically fails over to healthy resources in a user-defined sequence.
-   🩺 **Health Checks:** Actively monitors the status of all resources and provides a detailed history.
-   📧 **Failure Notifications:** Sends email alerts or triggers webhooks when a resource goes down.
-   🔧 **Dynamic Configuration:** Manage Resources, Endpoints, Clients, and Rules through a clean web interface.
-   🔒 **Secure:** All API endpoints are protected by client-specific authentication tokens.
-   🧪 **Development Mode:** A powerful mode for overriding incoming data for testing and development without changing client code.
-   🐳 **Containerized:** Easy to deploy and run anywhere with a simple `docker-compose up`.

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
    Open the `docker-compose.yml` file and change the `GATEWAY_ADMIN_PASSWORD` to a strong, secret password.

3.  **Build and Run:**
    Run the following command from the project's root directory. This will build the Docker image and start the container in the background.
    ```bash
    docker-compose up -d --build
    ```

4.  **Access the Gateway:**
    The application is now running!
    - **UI:** Open your browser to `http://localhost:8000`
    - **Default Password:** The password you set in the `docker-compose.yml` file.

---

## 📖 Functional Documentation

Dispatchapy is configured entirely through its web interface. Here are the core concepts:

### 1. Resources
A **Resource** is a destination API endpoint that you want to send data to. Each resource has:
- **Health Checks:** The gateway periodically sends a test `POST` request to the resource's endpoint using the "Health Check Values" you define. If the check fails, the resource is marked as "DOWN".
- **Notifications:** You can configure email and webhook notifications to be triggered when a resource's health status changes from UP to DOWN.

### 2. Endpoints
An **Endpoint** is a custom API route that you create on the gateway (e.g., `/api/send_alert`). This is the URL that your own applications will call. An endpoint is configured with:
- **Associated Resources:** A prioritised, ordered list of resources to be used for failover. The worker will always try the first healthy resource in the sequence.
- **Required Incoming Parameters:** The list of fields that a client application must provide in its JSON payload (e.g., `phone`, `message`).
- **Development Mode:** A toggle that enables special testing rules.

### 3. Clients & Authentication
Every call to a Dispatchapy endpoint must be authenticated.
- **Clients:** You can create multiple clients for each endpoint. Each client is given a unique, secret `X-API-Token`.
- **Authentication:** The client application must include its token in the `X-API-Token` HTTP header with every request.

### 4. Rules
The "Rules" page is where you define how data is transformed. It has two main parts:
- **Parameter Mapping:** This translates the data from your client's payload into the format required by a specific resource. For example, you can map an incoming `phone` field to the `recipient_number` field that your provider expects.
- **Development Rules:** When "Dev Mode" is on, these rules can override incoming data for testing. You can set default overrides (e.g., always send to a test phone number) or conditional overrides (e.g., if `scope` is "testing", send to a test number).

### 5. Task Lifecycle
- **Pending:** A task is created in this state. The worker will pick it up when a healthy resource is available.
- **Processing:** A worker has locked the task and is attempting to send it.
- **Success:** The notification was successfully sent.
- **Failed:** The task failed on all available resources for the maximum number of configured retries.

---

## 🛠️ Technical Documentation

### Tech Stack
- **Backend:** FastAPI (Python)
- **Database:** SQLite (via SQLAlchemy)
- **UI:** Jinja2 templates with Tailwind CSS and Alpine.js
- **Containerization:** Docker & Docker Compose

### Architecture
Dispatchapy runs on two main background processes (workers):
1.  **Task Execution Worker (`worker_loop`):** This is a high-frequency worker that constantly polls the database for `pending` tasks. It finds the next healthy resource in an endpoint's failover sequence and attempts to execute the task. It handles all retry and failover logic.
2.  **Health Check Worker (`resource_health_check_loop`):** This is a lower-frequency worker that periodically checks the status of all active resources. It is responsible for marking resources as `UP` or `DOWN` and triggering failure notifications.

---

## 💡 Potential Usage

While I developed this originally for managing notification services for my projects, Dispatchapy is a generic API dispatcher and can be used to add resilience and a unified interface to any API that accepts a `POST` request.

- **SMS Gateways:** Twilio, Vonage, etc.
- **Transactional Email Services:** SendGrid, Mailgun, Postmark.
- **Push Notification Services:** Firebase Cloud Messaging (FCM), OneSignal.
- **Internal Webhooks:** Sending reliable, retried notifications between your own microservices.
- **Team Chat Notifications:** Sending alerts to Slack, Discord, or Microsoft Teams.
- **IoT (Internet of Things):** Sending commands to smart devices.

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

