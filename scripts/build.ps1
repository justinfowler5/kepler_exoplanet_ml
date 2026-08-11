param(
    [string]$Tag = "kepler-engine:local",
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

docker buildx version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "docker buildx is required. Install/update Docker Desktop or the buildx plugin."
}

# Ensure a usable builder is selected (Docker Desktop's default is fine for --load).
$builder = (docker buildx ls 2>$null | Select-String -Pattern '\*' | Select-Object -First 1)
if (-not $builder) {
    Write-Host "No buildx builder selected; creating 'kepler' builder ..."
    docker buildx create --name kepler --driver docker-container --use | Out-Null
} else {
    Write-Host "Using buildx builder: $($builder.ToString().Trim())"
}

$env:TAG = $Tag
$bakeArgs = @("--progress=plain")
if ($NoCache) {
    $bakeArgs += "--no-cache"
}

Write-Host "Building $Tag via docker buildx bake ..."
# Pin the bake file so Compose is not also loaded as a bake definition.
docker buildx bake -f docker-bake.hcl @bakeArgs
if ($LASTEXITCODE -ne 0) {
    throw "docker buildx bake failed with exit code $LASTEXITCODE"
}

docker image inspect $Tag 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Image '$Tag' was not loaded into the local Docker Engine."
}

Write-Host "Built and loaded $Tag"
