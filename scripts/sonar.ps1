param(
    [Parameter(Position = 0)]
    [ValidateSet("up", "down", "coverage", "scan", "all")]
    [string]$Task = "all"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

$ContainerName = "kepler-sonarqube"
$SonarHostPort = 9002
$SonarUrl = "http://localhost:$SonarHostPort"
$SonarImage = "sonarqube:community"
$ScannerImage = "sonarsource/sonar-scanner-cli:11"
$AdminUser = "admin"
$AdminPassword = "AdminChangeMe1!"
$TokenName = "kepler-local-scanner"
$TokenFile = Join-Path (Join-Path $RepoRoot ".sonar") "token"
# Local Community project key (SonarCloud uses justinfowler5_kepler_exoplanet_ml via sonar-project.properties).
$LocalProjectKey = "kepler-engine"

function Wait-SonarReady {
    Write-Host "Waiting for SonarQube at $SonarUrl ..."
    $deadline = (Get-Date).AddMinutes(5)
    while ((Get-Date) -lt $deadline) {
        try {
            $status = Invoke-RestMethod -Uri "$SonarUrl/api/system/status" -TimeoutSec 5
            if ($status.status -eq "UP") {
                Write-Host "SonarQube is UP."
                return
            }
            Write-Host "  status=$($status.status)"
        } catch {
            Write-Host "  not ready yet..."
        }
        Start-Sleep -Seconds 5
    }
    throw "SonarQube did not become ready within 5 minutes."
}

function Ensure-SonarPassword {
    # /api/authentication/validate returns HTTP 200 with {valid:false} for bad creds.
    $newBasic = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("${AdminUser}:${AdminPassword}"))
    try {
        $check = Invoke-RestMethod -Uri "$SonarUrl/api/authentication/validate" `
            -Headers @{ Authorization = "Basic $newBasic" } `
            -TimeoutSec 10
        if ($check.valid -eq $true) {
            return
        }
    } catch {
        # Fall through and try the factory default.
    }

    $oldBasic = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("${AdminUser}:admin"))
    $body = @{
        login = $AdminUser
        previousPassword = "admin"
        password = $AdminPassword
    }
    try {
        Invoke-RestMethod -Method Post -Uri "$SonarUrl/api/users/change_password" `
            -Headers @{ Authorization = "Basic $oldBasic" } `
            -Body $body -TimeoutSec 15 | Out-Null
        Write-Host "Changed default admin password."
    } catch {
        Write-Host "Could not change admin password (may already be set): $($_.Exception.Message)"
        throw "Unable to authenticate to SonarQube as admin. Reset the container with 'down' then 'up'."
    }
}

function Get-OrCreate-Token {
    New-Item -ItemType Directory -Force -Path (Split-Path $TokenFile) | Out-Null
    if (Test-Path $TokenFile) {
        $existing = (Get-Content $TokenFile -Raw).Trim()
        if ($existing) { return $existing }
    }

    $pair = "${AdminUser}:${AdminPassword}"
    $basic = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($pair))
    $headers = @{ Authorization = "Basic $basic" }

    # Revoke a prior token with the same name so regenerate is idempotent.
    try {
        Invoke-RestMethod -Method Post -Uri "$SonarUrl/api/user_tokens/revoke" `
            -Headers $headers -Body @{ name = $TokenName } -TimeoutSec 15 | Out-Null
    } catch {
        # Token may not exist yet.
    }

    $resp = Invoke-RestMethod -Method Post -Uri "$SonarUrl/api/user_tokens/generate" `
        -Headers $headers -Body @{ name = $TokenName } -TimeoutSec 15
    $token = $resp.token
    Set-Content -Path $TokenFile -Value $token -NoNewline
    Write-Host "Created scanner token -> $TokenFile"
    return $token
}

function Start-Sonar {
    $existing = docker ps -a --filter "name=^/${ContainerName}$" --format "{{.Names}}"
    if ($existing) {
        $running = docker ps --filter "name=^/${ContainerName}$" --format "{{.Names}}"
        if (-not $running) {
            Write-Host "Starting existing container $ContainerName ..."
            docker start $ContainerName | Out-Null
        } else {
            Write-Host "Container $ContainerName already running."
        }
    } else {
        Write-Host "Pulling and starting $SonarImage on host port $SonarHostPort ..."
        docker run -d `
            --name $ContainerName `
            -p "${SonarHostPort}:9000" `
            -e SONAR_ES_BOOTSTRAP_CHECKS_DISABLE=true `
            $SonarImage | Out-Null
    }
    Wait-SonarReady
    Ensure-SonarPassword
    Get-OrCreate-Token | Out-Null
    Write-Host "UI: $SonarUrl  (login: $AdminUser / $AdminPassword)"
}

function Stop-Sonar {
    docker rm -f $ContainerName 2>$null | Out-Null
    Write-Host "Removed container $ContainerName."
}

function Invoke-Coverage {
    Write-Host "Running pytest with coverage ..."
    uv sync --group dev
    uv run pytest --cov=kepler_engine --cov-report=xml:coverage.xml --cov-report=term-missing -q
    if (-not (Test-Path "coverage.xml")) {
        throw "coverage.xml was not produced."
    }
    Write-Host "Wrote coverage.xml"
}

function Invoke-Scan {
    if (-not (Test-Path "coverage.xml")) {
        throw "coverage.xml missing. Run coverage first (or use -Task all)."
    }
    if (-not (Test-Path $TokenFile)) {
        throw "Scanner token missing at $TokenFile. Run 'up' first."
    }
    $token = (Get-Content $TokenFile -Raw).Trim()

    Write-Host "Running sonar-scanner-cli against $SonarUrl ..."
    docker run --rm `
        -e SONAR_HOST_URL="http://host.docker.internal:$SonarHostPort" `
        -e SONAR_TOKEN="$token" `
        -v "${RepoRoot}:/usr/src" `
        $ScannerImage `
        -Dsonar.projectKey=$LocalProjectKey

    Write-Host ""
    Write-Host "Open $SonarUrl/dashboard?id=$LocalProjectKey for bugs, smells, security, and coverage."
    Write-Host "GitHub PR decoration uses SonarCloud via .github/workflows/sonarcloud.yml (not this local instance)."
}

switch ($Task) {
    "up" { Start-Sonar }
    "down" { Stop-Sonar }
    "coverage" { Invoke-Coverage }
    "scan" {
        Wait-SonarReady
        Invoke-Scan
    }
    "all" {
        Start-Sonar
        Invoke-Coverage
        Invoke-Scan
    }
}
