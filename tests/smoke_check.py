# Check tasks and task logs for the smoke test endpoint
from dispatchapy import SessionLocal, Task, TaskLog, Endpoint, Client

db = SessionLocal()
try:
    tasks = db.query(Task).join(Endpoint, Task.endpoint_id==Endpoint.id).filter(Endpoint.path=='smoke_test_ep').order_by(Task.created_at.desc()).all()
    for t in tasks[:5]:
        print('Task', t.id, 'status', t.status, 'created_at', t.created_at)
        logs = db.query(TaskLog).filter(TaskLog.task_id==t.id).order_by(TaskLog.created_at).all()
        for l in logs:
            print('  LOG:', l.created_at, l.message)
finally:
    db.close()
