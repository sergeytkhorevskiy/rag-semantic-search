.PHONY: help build run stop clean logs shell test dev prod

# Default target
help:
	@echo "Available commands:"
	@echo "  build    - Build Docker image"
	@echo "  run      - Run the application"
	@echo "  stop     - Stop the application"
	@echo "  clean    - Clean up containers and images"
	@echo "  logs     - Show application logs"
	@echo "  shell    - Open shell in running container"
	@echo "  test     - Run tests"
	@echo "  dev      - Run in development mode"
	@echo "  prod     - Run in production mode"
	@echo "  ingest   - Run document ingestion (original)"
	@echo "  ingest-parallel - Run parallel document ingestion"
	@echo "  ingest-async - Run async document ingestion"
	@echo "  rebuild  - Rebuild and restart"

# Build the Docker image
build:
	docker-compose build

# Run the application
run:
	docker-compose up -d

# Stop the application
stop:
	docker-compose down

# Clean up containers and images
clean:
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f

# Show application logs
logs:
	docker-compose logs -f

# Open shell in running container
shell:
	docker-compose exec rag-server /bin/bash

# Run tests
test:
	docker-compose run --rm rag-server python -m pytest

# Development mode with hot reload
dev:
	docker-compose up -d

# Production mode
prod:
	docker-compose up -d

# Run document ingestion (original)
ingest:
	docker-compose run --rm rag-server python scripts/ingest.py

# Run parallel document ingestion
ingest-parallel:
	docker-compose run --rm rag-server python scripts/ingest_parallel.py

# Run async document ingestion
ingest-async:
	docker-compose run --rm rag-server python scripts/ingest_async.py

# Run ingestion with performance monitoring
ingest-benchmark:
	@echo "=== Benchmarking different ingest methods ==="
	@echo "1. Original ingest:"
	@time docker-compose run --rm rag-server python scripts/ingest.py
	@echo "2. Parallel ingest:"
	@time docker-compose run --rm rag-server python scripts/ingest_parallel.py
	@echo "3. Async ingest:"
	@time docker-compose run --rm rag-server python scripts/ingest_async.py

# Rebuild and restart
rebuild: clean build run

# Health check
health:
	curl -f http://localhost:8000/health || echo "API server is not healthy"
	curl -f http://localhost:7860/ || echo "UI server is not healthy"

# Scale services
scale:
	docker-compose up -d --scale rag-server=3

# Performance monitoring
perf:
	@echo "=== Performance Information ==="
	@echo "CPU cores: $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")"
	@echo "Memory: $(shell free -h 2>/dev/null || vm_stat 2>/dev/null || echo "unknown")"
	@echo "Docker containers:"
	docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
