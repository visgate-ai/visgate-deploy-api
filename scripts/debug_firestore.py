from google.cloud import firestore
import os

os.environ["GCP_PROJECT_ID"] = "visgate"
client = firestore.Client(project="visgate")
collection = "deployments"

print(f"Sampling documents from '{collection}'...")
docs = client.collection(collection).limit(5).stream()

for doc in docs:
    print(f"ID: {doc.id}, Data: {doc.to_dict()}")
