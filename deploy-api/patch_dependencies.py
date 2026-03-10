import re

with open("src/api/dependencies.py", "r") as f:
    text = f.read()

# Make get_firestore_client return to dependencies for the tests mocking it.

text += "\nfrom src.services.db import get_firestore_client\n"

with open("src/api/dependencies.py", "w") as f:
    f.write(text)

