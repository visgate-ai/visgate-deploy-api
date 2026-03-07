from google.cloud import firestore
import os

os.environ['GCP_PROJECT_ID'] = 'visgate'
db = firestore.Client()

gpu_registry = [
    {"id": "NVIDIA RTX A4000",          "display": "NVIDIA A4000",          "vram": 16, "cost_index": 1},
    {"id": "NVIDIA RTX A5000",          "display": "NVIDIA A5000",          "vram": 24, "cost_index": 2},
    {"id": "NVIDIA GeForce RTX 3090",   "display": "NVIDIA RTX 3090",       "vram": 24, "cost_index": 2},
    {"id": "NVIDIA GeForce RTX 4090",   "display": "NVIDIA RTX 4090",       "vram": 24, "cost_index": 3},
    {"id": "NVIDIA RTX A6000",          "display": "NVIDIA A6000",          "vram": 48, "cost_index": 5},
    {"id": "NVIDIA L40S",               "display": "NVIDIA L40S",           "vram": 48, "cost_index": 6},
    {"id": "NVIDIA A100 80GB PCIe",     "display": "NVIDIA A100 (80GB)",    "vram": 80, "cost_index": 8},
    {"id": "NVIDIA H100 PCIe",          "display": "NVIDIA H100",           "vram": 80, "cost_index": 10},
]

tier_mapping = {
    "ECONOMY": ["NVIDIA RTX A4000", "NVIDIA RTX A5000"],
    "STANDARD": ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"],
    "PRO": ["NVIDIA RTX A6000", "NVIDIA L40S"],
    "ULTIMATE": ["NVIDIA A100 80GB PCIe", "NVIDIA H100 PCIe"],
}

print("Populating GPU registry...")
for gpu in gpu_registry:
    db.collection("visgate_deploy_api_gpu_registry").document(gpu["id"]).set(gpu)
    print(f"Added {gpu['id']}")

print("\nPopulating GPU tiers...")
for tier, gpu_ids in tier_mapping.items():
    db.collection("visgate_deploy_api_gpu_tiers").document(tier).set({"gpu_ids": gpu_ids})
    print(f"Added tier {tier}")

print("\nDone!")
