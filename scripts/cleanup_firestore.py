import hashlib
import os
from google.cloud import firestore

# Determine user_hash from .env.local
runpod_key = ""
if os.path.isfile(".env.local"):
    with open(".env.local") as f:
        for line in f:
            if line.startswith("RUNPOD="):
                runpod_key = line.split("=")[1].strip().strip('"')
                break

if not runpod_key:
    # Try alternate name
    if os.path.isfile(".env.local"):
        with open(".env.local") as f:
            for line in f:
                if line.startswith("RUNPOD_API_KEY="):
                    runpod_key = line.split("=")[1].strip().strip('"')
                    break

if not runpod_key:
    print("Error: RUNPOD API key not found in .env.local")
    exit(1)

user_hash = hashlib.sha256(runpod_key.encode("utf-8")).hexdigest()
print(f"User hash: {user_hash}")

# Initialize Firestore
os.environ["GCP_PROJECT_ID"] = "visgate" # Hardcoded based on my earlier check
client = firestore.Client(project="visgate")
collection = "deployments" # Default from config

print(f"Cleaning up Firestore collection '{collection}' for user {user_hash}...")

query = client.collection(collection).where("user_hash", "==", user_hash).stream()

count = 0
for doc in query:
    data = doc.to_dict()
    if data.get("status") != "deleted":
        print(f"  Marking {doc.id} as deleted")
        doc.reference.update({"status": "deleted"})
        count += 1

print(f"Done. Marked {count} deployments as deleted.")
