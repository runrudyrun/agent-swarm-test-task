#!/bin/bash
set -e

echo "🚀 Starting InfinitePay Agent Swarm API..."

# Wait for environment to be ready
sleep 2

# Seed the mounted /app/data from baked assets if it's empty
echo "📦 Checking for baked index to seed /app/data if empty..."

# Ensure base directories exist
mkdir -p /app/data/chroma /app/data/raw || true

# If chroma DB file is missing on the mounted disk but exists in baked data, copy it over
if [ ! -e "/app/data/chroma/chroma.sqlite3" ] && [ -d "/opt/baked_data/chroma" ]; then
  echo "➡️  Seeding Chroma index from /opt/baked_data/chroma"
  cp -a /opt/baked_data/chroma/. /app/data/chroma/ || true
fi

# If no raw JSON exists on the mounted disk but exists in baked data, copy it too (useful for reindex)
if [ -d "/opt/baked_data/raw" ]; then
  if [ -z "$(ls -A /app/data/raw 2>/dev/null || true)" ]; then
    echo "➡️  Seeding raw scraped data from /opt/baked_data/raw"
    cp -a /opt/baked_data/raw/. /app/data/raw/ || true
  fi
fi

# Relaxed: fix ownership when possible (non-fatal)
chown -R appuser:appuser /app/data 2>/dev/null || true

# Run startup checks using existing Python infrastructure
echo "🔍 Checking vector store configuration..."
python /app/docker_startup_check.py

echo "🎯 Starting API server..."
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 "$@"