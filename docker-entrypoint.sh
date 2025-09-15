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
  # Do not preserve ownership/permissions to avoid read-only DB issues on mounted volumes
  cp -r /opt/baked_data/chroma/. /app/data/chroma/ || true
fi

# If no raw JSON exists on the mounted disk but exists in baked data, copy it too (useful for reindex)
if [ -d "/opt/baked_data/raw" ]; then
  if [ -z "$(ls -A /app/data/raw 2>/dev/null || true)" ]; then
    echo "➡️  Seeding raw scraped data from /opt/baked_data/raw"
    cp -r /opt/baked_data/raw/. /app/data/raw/ || true
  fi
fi

# Ensure current runtime user owns the mounted data directory when possible
chown -R "$(id -u)":"$(id -g)" /app/data 2>/dev/null || true

# Normalize permissions so the running user can write the SQLite DB
chmod -R u+rwX,g+rwX /app/data 2>/dev/null || true
chmod -R o+rwx /app/data 2>/dev/null || true

# As a last resort for PaaS volumes with restrictive ownership, make data dir world-writable
chmod -R 777 /app/data 2>/dev/null || true

# Ensure permissive umask so newly created files are writable by all if needed
umask 000

# If the chroma DB file exists but is not writable, attempt to fix or remove it
if [ -e "/app/data/chroma/chroma.sqlite3" ] && [ ! -w "/app/data/chroma/chroma.sqlite3" ]; then
  echo "🛠️  chroma.sqlite3 is not writable; attempting to fix permissions"
  chmod u+rw,g+rw "/app/data/chroma/chroma.sqlite3" 2>/dev/null || true
  if [ ! -w "/app/data/chroma/chroma.sqlite3" ]; then
    echo "🧹 Removing read-only chroma.sqlite3 to allow re-creation"
    rm -f "/app/data/chroma/chroma.sqlite3" 2>/dev/null || true
  fi
fi

# Run startup checks using existing Python infrastructure
echo "🔍 Checking vector store configuration..."
python /app/docker_startup_check.py

echo "🎯 Starting API server..."
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 "$@"