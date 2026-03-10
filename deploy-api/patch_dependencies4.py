import re
with open("tests/unit/test_deployment_creds.py", "r") as f:
    text = f.read()

text = text.replace('patch("src.services.deployment.get_firestore_client",', 'patch("src.services.db.get_firestore_client",')

with open("tests/unit/test_deployment_creds.py", "w") as f:
    f.write(text)

with open("tests/unit/test_tasks.py", "r") as f:
    text = f.read()

text = text.replace('monkeypatch.setattr(dependencies, "get_firestore_client",', 'monkeypatch.setattr("src.services.db.get_firestore_client",')

with open("tests/unit/test_tasks.py", "w") as f:
    f.write(text)

