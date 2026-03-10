import requests
import os

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("Missing RUNPOD_API_KEY")
    exit(1)

def runpod_graphql(query: str, variables: dict = None):
    url = "https://api.runpod.io/graphql"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    resp = requests.post(url, json={"query": query, "variables": variables or {}}, headers=headers)
    resp.raise_for_status()
    return resp.json()

def get_all_endpoints():
    q = """
    query {
        endpoints {
            id
            name
        }
    }
    """
    res = runpod_graphql(q)
    return res.get("data", {}).get("endpoints", [])

def main():
    endpoints = get_all_endpoints()
    print(f"Found {len(endpoints)} endpoints in RunPod.")
    # In a real scenario we'd match these with Firestore to find zombies.
    print("Zombie reaper run complete.")

if __name__ == "__main__":
    main()
