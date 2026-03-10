import re

with open("src/api/dependencies.py", "r") as f:
    text = f.read()

# Make get_firestore_client return to dependencies for the tests mocking it.

text = text.replace("from google.cloud import firestore", "from google.cloud import firestore\nfrom src.services.db import get_firestore_client")
text = text.replace("def get_firestore():\n    \"\"\"Return Firestore client for current project.\"\"\"\n    settings = get_settings()\n    return _get_repo().get_firestore_client(settings.gcp_project_id)", "def get_firestore():\n    \"\"\"Return Firestore client for current project.\"\"\"\n    settings = get_settings()\n    return get_firestore_client(settings.gcp_project_id)")

with open("src/api/dependencies.py", "w") as f:
    f.write(text)

