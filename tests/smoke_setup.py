"""Insert a test endpoint, client, and a client dev rule using the app's models."""
from dispatchapy import SessionLocal, Endpoint, Client, ClientDevRule, EndpointClientLink, engine, run_auto_migrations

# Ensure DB schema
try:
    run_auto_migrations(engine)
except Exception:
    pass

db = SessionLocal()
try:
    # clean up any prior test artifacts
    db.query(EndpointClientLink).filter(EndpointClientLink.client_id == None).delete()
    db.commit()

    # create endpoint
    ep = db.query(Endpoint).filter(Endpoint.path == 'smoke_test_ep').first()
    if not ep:
        ep = Endpoint(path='smoke_test_ep', description='Smoke test endpoint', required_params=['ref','scope','code'], dev_hold_tasks=False, is_development_mode=False)
        db.add(ep)
        db.commit()
        db.refresh(ep)

    # create client
    client = db.query(Client).filter(Client.token == 'smoke-token').first()
    if not client:
        client = Client(name='smoke-client', token='smoke-token', description='smoke client')
        db.add(client)
        db.commit()
        db.refresh(client)

    # link client to endpoint
    exists = db.query(EndpointClientLink).filter(EndpointClientLink.endpoint_id == ep.id, EndpointClientLink.client_id == client.id).first()
    if not exists:
        link = EndpointClientLink(endpoint_id=ep.id, client_id=client.id)
        db.add(link)
        db.commit()

    # add client dev rule: if payload.code matches 'ALPHA' -> hold task
    # remove existing rules for clarity
    db.query(ClientDevRule).filter(ClientDevRule.client_id == client.id).delete()
    db.commit()
    rule = ClientDevRule(client_id=client.id, endpoint_id=ep.id, condition_param='code', condition_value='ALPHA', target_param=None, override_value=None, active=True)
    db.add(rule)
    db.commit()
    print('Setup complete: endpoint id', ep.id, 'client id', client.id, 'rule id', rule.id)
finally:
    db.close()
