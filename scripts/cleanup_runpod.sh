#!/bin/bash
# Cleanup Runpod endpoints using curl and jq

RUNPOD_API_KEY=$(grep '^RUNPOD=' .env.local | cut -d'=' -f2)

if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Error: RUNPOD API key not found in .env.local"
    exit 1
fi

echo "Fetching endpoints..."
ENDPOINTS=$(curl -s -X POST -H "Content-Type: application/json" "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -d '{"query": "query { myself { endpoints { id name } } }"}' | jq -r '.data.myself.endpoints[] | select(.name | startswith("visgate-")) | .id')

if [ -z "$ENDPOINTS" ]; then
    echo "No matching endpoints found."
    exit 0
fi

for id in $ENDPOINTS; do
    echo "Deleting endpoint: $id"
    curl -s -X POST -H "Content-Type: application/json" "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -d "{\"query\": \"mutation { deleteEndpoint(id: \\\"$id\\\") }\"}"
done

echo "Cleanup complete."
