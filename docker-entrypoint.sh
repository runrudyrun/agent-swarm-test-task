#!/bin/bash
set -e

echo "🚀 Starting InfinitePay Agent Swarm API..."

# Wait for environment to be ready
sleep 2

# Run startup checks using existing Python infrastructure
echo "🔍 Checking vector store configuration..."
python /app/docker_startup_check.py

echo "🎯 Starting API server..."
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 "$@"