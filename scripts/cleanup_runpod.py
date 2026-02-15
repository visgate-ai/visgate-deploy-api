import os
import requests
import json

RUNPOD_API_KEY = os.environ.get("RUNPOD") or os.environ.get("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    try:
        with open(".env.local") as f:
            for line in f:
                if line.startswith("RUNPOD="):
                    RUNPOD_API_KEY = line.split("=")[1].strip().strip('"')
                    break
    except FileNotFoundError:
        pass

if not RUNPOD_API_KEY:
    print("Error: RUNPOD API key not found in environment or .env.local")
    exit(1)

GRAPHQL_URL = f"https://api.runpod.io/graphql?api_key={RUNPOD_API_KEY}"

def get_endpoints():
    query = """
    query {
        myself {
            endpoints {
                id
                name
            }
        }
    }
    """
    resp = requests.post(GRAPHQL_URL, json={"query": query})
    if resp.status_code != 200:
        print(f"Error fetching endpoints: {resp.text}")
        return []
    data = resp.json()
    return data.get("data", {}).get("myself", {}).get("endpoints", [])

def delete_endpoint(endpoint_id):
    query = """
    mutation DeleteEndpoint($id: String!) {
        deleteEndpoint(id: $id)
    }
    """
    resp = requests.post(GRAPHQL_URL, json={"query": query, "variables": {"id": endpoint_id}})
    if resp.status_code != 200:
        print(f"Error deleting endpoint {endpoint_id}: {resp.text}")
    else:
        print(f"Deleted endpoint: {endpoint_id}")

if __name__ == "__main__":
    print("Fetching endpoints...")
    endpoints = get_endpoints()
    print(f"Found {len(endpoints)} endpoints.")
    
    for ep in endpoints:
        if ep["name"] and ep["name"].startswith("visgate-"):
            print(f"Deleting endpoint: {ep['name']} ({ep['id']})")
            delete_endpoint(ep["id"])
        else:
            print(f"Skipping endpoint: {ep['name']} ({ep['id']})")
    print("Cleanup complete.")
