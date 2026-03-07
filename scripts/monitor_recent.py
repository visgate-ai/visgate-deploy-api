from google.cloud import firestore
import os

os.environ["GCP_PROJECT_ID"] = "visgate"
client = firestore.Client(project="visgate")

print("Checking for recent deployments across all organizations...")
orgs = client.collection("organizations").stream()

recent_deps = []

for org in orgs:
    deps = client.collection("organizations").document(org.id).collection("deployments").order_by("created_at", direction=firestore.Query.DESCENDING).limit(1).stream()
    for d in deps:
        data = d.to_dict()
        recent_deps.append({
            "org_id": org.id,
            "dep_id": d.id,
            "status": data.get("status"),
            "created_at": data.get("created_at"),
            "model": data.get("model_id") or data.get("hf_model_id")
        })

# Sort by created_at
recent_deps.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)

for rd in recent_deps[:3]:
    print(f"Org: {rd['org_id']}, Dep: {rd['dep_id']}, Status: {rd['status']}, Created: {rd['created_at']}, Model: {rd['model']}")
