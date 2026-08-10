param(
    [Parameter(Position = 0)]
    [ValidateSet("image", "fs", "config", "all")]
    [string]$Task = "image",

    [string]$ImageTag = "kepler-engine:local",

    [ValidateSet("UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL")]
    [string]$Severity = "HIGH",

    [switch]$SkipBuild,
    [switch]$FailOnFindings
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

$TrivyImage = "aquasec/trivy:0.67.2"
$ReportsDir = Join-Path (Join-Path $RepoRoot "reports") "trivy"
$DockerSock = "//var/run/docker.sock"

function Get-SeverityList {
    $order = @("UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL")
    $idx = [array]::IndexOf($order, $Severity)
    if ($idx -lt 0) { return "HIGH,CRITICAL" }
    return ($order[$idx..($order.Length - 1)] -join ",")
}

function Ensure-ReportsDir {
    New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
}

function Invoke-Trivy {
    param([Parameter(Mandatory = $true)][string[]]$TrivyArgs)

    docker run --rm `
        -v "${DockerSock}:/var/run/docker.sock" `
        -v "${RepoRoot}:/workspace" `
        -w /workspace `
        $TrivyImage @TrivyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "trivy exited with code $LASTEXITCODE"
    }
}

function Build-AppImage {
    if ($SkipBuild) {
        docker image inspect $ImageTag 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Image '$ImageTag' not found. Omit -SkipBuild or build it first."
        }
        Write-Host "Using existing image $ImageTag"
        return
    }
    Write-Host "Building $ImageTag from Dockerfile ..."
    docker build -t $ImageTag .
    if ($LASTEXITCODE -ne 0) {
        throw "docker build failed with code $LASTEXITCODE"
    }
}

function Scan-Image {
    Ensure-ReportsDir
    Build-AppImage

    $sev = Get-SeverityList
    $tableOut = "reports/trivy/image.txt"
    $jsonOut = "reports/trivy/image.json"
    $sarifOut = "reports/trivy/image.sarif"

    Write-Host "Trivy image scan ($ImageTag), severity >= $Severity ($sev) ..."

    $common = @(
        "image"
        "--severity", $sev
        "--ignore-unfixed"
    )

    Invoke-Trivy ($common + @("--format", "table", "--output", $tableOut, $ImageTag))
    Invoke-Trivy ($common + @("--format", "json", "--output", $jsonOut, $ImageTag))
    Invoke-Trivy ($common + @("--format", "sarif", "--output", $sarifOut, $ImageTag))

    if ($FailOnFindings) {
        Invoke-Trivy ($common + @("--exit-code", "1", "--format", "table", $ImageTag))
    }

    Write-Host ""
    Get-Content (Join-Path $RepoRoot $tableOut)
    Write-Host ""
    Write-Host "Reports: $tableOut | $jsonOut | $sarifOut"
}

function Scan-Filesystem {
    Ensure-ReportsDir
    $sev = Get-SeverityList
    $tableOut = "reports/trivy/fs.txt"
    $jsonOut = "reports/trivy/fs.json"

    Write-Host "Trivy filesystem scan (repo root), severity >= $Severity ..."
    $common = @(
        "fs"
        "--severity", $sev
        "--scanners", "vuln,secret,misconfig"
        "--ignore-unfixed"
    )
    Invoke-Trivy ($common + @("--format", "table", "--output", $tableOut, "."))
    Invoke-Trivy ($common + @("--format", "json", "--output", $jsonOut, "."))

    Write-Host ""
    Get-Content (Join-Path $RepoRoot $tableOut)
    Write-Host ""
    Write-Host "Reports: $tableOut | $jsonOut"
}

function Scan-Config {
    Ensure-ReportsDir
    $sev = Get-SeverityList
    $tableOut = "reports/trivy/config.txt"
    $jsonOut = "reports/trivy/config.json"

    Write-Host "Trivy config/IaC scan (Dockerfile + compose) ..."
    $common = @("config", "--severity", $sev)
    Invoke-Trivy ($common + @("--format", "table", "--output", $tableOut, "."))
    Invoke-Trivy ($common + @("--format", "json", "--output", $jsonOut, "."))

    Write-Host ""
    Get-Content (Join-Path $RepoRoot $tableOut)
    Write-Host ""
    Write-Host "Reports: $tableOut | $jsonOut"
}

switch ($Task) {
    "image" { Scan-Image }
    "fs" { Scan-Filesystem }
    "config" { Scan-Config }
    "all" {
        Scan-Image
        Scan-Filesystem
        Scan-Config
    }
}
