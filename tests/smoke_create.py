# Simple setup script to create a test endpoint, client, link, and client dev rule
from dispatchapy import SessionLocal, Endpoint, Client, EndpointClientLink, ClientDevRule
from dispatchapy import engine

db = SessionLocal()
try:
    # Clean up any existing test rows
    db.query(EndpointClientLink).filter(EndpointClientLink.endpoint_id.in_(db.query(Endpoint.id).filter(Endpoint.path=='smoke_test_ep'))).delete(synchronize_session=False)
    db.query(ClientDevRule).filter(ClientDevRule.client_id.in_(db.query(Client.id).filter(Client.name=='smoke_client'))).delete(synchronize_session=False)
    db.query(Endpoint).filter(Endpoint.path=='smoke_test_ep').delete(synchronize_session=False)
    db.query(Client).filter(Client.name=='smoke_client').delete(synchronize_session=False)
    db.commit()

    # Create endpoint
    ep = Endpoint(path='smoke_test_ep', description='smoke test', required_params=['ref','scope'], dev_hold_tasks=False)
    db.add(ep)
    db.commit()
    db.refresh(ep)

    # Create client
    client = Client(name='smoke_client', token='smoke_token_123', description='smoke client')
    db.add(client)
    db.commit()
    db.refresh(client)

    # Link endpoint and client
    link = EndpointClientLink(endpoint_id=ep.id, client_id=client.id)
    db.add(link)
    db.commit()

    # Add client dev rule that matches scope == 'smoke' (simple substring regex)
    r = ClientDevRule(client_id=client.id, endpoint_id=ep.id, condition_param='scope', condition_value='^smoke$', target_param=None, override_value=None, active=True)
    db.add(r)
    db.commit()
    print('Setup done: endpoint id', ep.id, 'client id', client.id, 'rule id', r.id)
finally:
    db.close()
