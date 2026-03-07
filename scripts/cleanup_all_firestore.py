from google.cloud import firestore
import os

os.environ["GCP_PROJECT_ID"] = "visgate"
client = firestore.Client(project="visgate")

print("Listing organizations...")
orgs = client.collection("organizations").stream()

for org in orgs:
    print(f"Org ID: {org.id}")
    deployments = client.collection("organizations").document(org.id).collection("deployments").stream()
    for d in deployments:
        data = d.to_dict()
        if data.get("status") != "deleted":
            print(f"  Deleting deployment {d.id} ({data.get('status')})")
            d.reference.update({"status": "deleted"})
