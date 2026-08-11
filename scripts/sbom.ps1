param(
    [Parameter(Position = 0)]
    [ValidateSet("image", "project", "all", "scan")]
    [string]$Task = "all",

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
$SbomDir = Join-Path (Join-Path $RepoRoot "reports") "sbom"
$DockerSock = "//var/run/docker.sock"

function Get-SeverityList {
    $order = @("UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL")
    $idx = [array]::IndexOf($order, $Severity)
    if ($idx -lt 0) { return "HIGH,CRITICAL" }
    return ($order[$idx..($order.Length - 1)] -join ",")
}

function Ensure-SbomDir {
    New-Item -ItemType Directory -Force -Path $SbomDir | Out-Null
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
    Write-Host "Building $ImageTag via buildx ..."
    & (Join-Path $PSScriptRoot "build.ps1") -Tag $ImageTag
}

function New-ImageSbom {
    Ensure-SbomDir
    Build-AppImage

    $cdx = "reports/sbom/kepler-engine.cdx.json"
    $spdx = "reports/sbom/kepler-engine.spdx.json"

    Write-Host "Generating CycloneDX SBOM for $ImageTag ..."
    Invoke-Trivy @("image", "--format", "cyclonedx", "--output", $cdx, $ImageTag)

    Write-Host "Generating SPDX SBOM for $ImageTag ..."
    Invoke-Trivy @("image", "--format", "spdx-json", "--output", $spdx, $ImageTag)

    Write-Host "Image SBOMs: $cdx | $spdx"
}

function New-ProjectSbom {
    Ensure-SbomDir

    $cdx = "reports/sbom/project.cdx.json"
    $spdx = "reports/sbom/project.spdx.json"
    # Skip local env / caches so the SBOM reflects lockfile + source, not installed site-packages.
    $skipDirs = ".venv,.local-stack,.pytest_cache,.ruff_cache,.scannerwork,reports,mlruns,htmlcov,__pycache__"

    Write-Host "Generating CycloneDX SBOM for project filesystem (uv.lock / Python deps) ..."
    Invoke-Trivy @(
        "fs"
        "--format", "cyclonedx"
        "--output", $cdx
        "--skip-dirs", $skipDirs
        "."
    )

    Write-Host "Generating SPDX SBOM for project filesystem ..."
    Invoke-Trivy @(
        "fs"
        "--format", "spdx-json"
        "--output", $spdx
        "--skip-dirs", $skipDirs
        "."
    )

    Write-Host "Project SBOMs: $cdx | $spdx"
}

function Scan-OneSbom {
    param(
        [Parameter(Mandatory = $true)][string]$SbomPath,
        [Parameter(Mandatory = $true)][string]$ReportStem
    )

    if (-not (Test-Path $SbomPath)) {
        throw "SBOM not found: $SbomPath - run .\scripts\sbom.ps1 image|project|all first."
    }

    $sev = Get-SeverityList
    $tableOut = "reports/sbom/${ReportStem}-scan.txt"
    $jsonOut = "reports/sbom/${ReportStem}-scan.json"

    Write-Host "Scanning SBOM $SbomPath (severity >= $Severity) ..."
    $common = @(
        "sbom"
        "--severity", $sev
        "--ignore-unfixed"
        "--scanners", "vuln"
    )

    Invoke-Trivy ($common + @("--format", "table", "--output", $tableOut, $SbomPath))
    Invoke-Trivy ($common + @("--format", "json", "--output", $jsonOut, $SbomPath))

    if ($FailOnFindings) {
        Invoke-Trivy ($common + @("--exit-code", "1", "--format", "table", $SbomPath))
    }

    Write-Host ""
    Get-Content (Join-Path $RepoRoot $tableOut)
    Write-Host ""
    Write-Host ("Scan reports: {0} | {1}" -f $tableOut, $jsonOut)
}

function Scan-Sboms {
    Ensure-SbomDir
    Scan-OneSbom -SbomPath "reports/sbom/kepler-engine.cdx.json" -ReportStem "kepler-engine"
    Scan-OneSbom -SbomPath "reports/sbom/project.cdx.json" -ReportStem "project"
}

switch ($Task) {
    "image" { New-ImageSbom }
    "project" { New-ProjectSbom }
    "all" {
        New-ImageSbom
        New-ProjectSbom
    }
    "scan" { Scan-Sboms }
}

Write-Host ""
Write-Host "SBOMs/reports are local-only under reports/sbom/ (gitignored). Do not commit them to the public repo."
