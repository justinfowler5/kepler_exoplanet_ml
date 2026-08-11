param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "run", "worker", "test", "lint", "build", "up", "down", "download", "sonar", "trivy", "sbom")]
    [string]$Task = "setup"
)

$ErrorActionPreference = "Stop"
$env:Path = "C:\Users\Justi\.local\bin;$env:Path"
$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"

switch ($Task) {
    "setup" {
        uv python pin 3.12
        uv sync
        if (-not (Test-Path .env)) { Copy-Item .env.example .env }
        Write-Host "Environment ready."
    }
    "run" {
        uv run uvicorn kepler_engine.main:app --reload --host 0.0.0.0 --port 8000
    }
    "worker" {
        uv run celery -A kepler_engine.workers.celery_app worker --loglevel=info --pool=solo
    }
    "test" {
        uv run pytest -q
    }
    "lint" {
        uv run ruff check .
    }
    "build" {
        & "$PSScriptRoot\scripts\build.ps1"
    }
    "up" {
        if (-not (Test-Path .env)) { Copy-Item .env.example .env }
        & "$PSScriptRoot\scripts\build.ps1"
        docker compose up -d --no-build
    }
    "down" {
        docker compose down -v
    }
    "download" {
        uv run python scripts/download_koi_dataset.py
    }
    "sonar" {
        & "$PSScriptRoot\scripts\sonar.ps1" all
    }
    "trivy" {
        & "$PSScriptRoot\scripts\trivy.ps1" image
    }
    "sbom" {
        & "$PSScriptRoot\scripts\sbom.ps1" all
    }
}
