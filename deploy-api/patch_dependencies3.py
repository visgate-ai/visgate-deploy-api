with open("tests/conftest.py", "r") as f:
    text = f.read()

# Mock settings directly
text = text.replace('monkeypatch.setattr(dependencies, "get_firestore_client", lambda project_id=None: mock_client)', 'monkeypatch.setattr("src.services.db.get_firestore_client", lambda project_id=None: mock_client)')

with open("tests/conftest.py", "w") as f:
    f.write(text)

