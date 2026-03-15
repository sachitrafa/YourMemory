#!/bin/bash
set -e

echo "YourMemory setup"
echo "----------------"

# Check Docker
if ! command -v docker &>/dev/null; then
  echo "Error: Docker is not installed. Install it from https://docs.docker.com/get-docker/"
  exit 1
fi

# Check Ollama
if ! command -v ollama &>/dev/null; then
  echo "Error: Ollama is not installed. Install it from https://ollama.com"
  exit 1
fi

# Pull embedding model
echo "Pulling Ollama embedding model (nomic-embed-text)..."
ollama pull nomic-embed-text

# Install Python package
echo "Installing cognitive-ai-memory..."
pip install cognitive-ai-memory

# Create .env if missing
if [ ! -f .env ]; then
  cat > .env <<EOF
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/yourmemory
OLLAMA_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
EXTRACT_MODEL=llama3.2:3b
EOF
  echo ".env created"
fi

# Start Postgres via Docker Compose
echo "Starting Postgres..."
docker compose up db -d

echo ""
echo "Done. Run 'yourmemory' to start the MCP server."
