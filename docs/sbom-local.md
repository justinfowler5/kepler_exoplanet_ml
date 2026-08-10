# Local SBOM generation (free)

Uses **Aqua Trivy** (already in this repo’s local tooling) to emit industry-standard SBOMs. No paid license.

Other free options (not wired here): [Syft](https://github.com/anchore/syft), [cdxgen](https://github.com/CycloneDX/cdxgen), [CycloneDX Python](https://github.com/CycloneDX/cyclonedx-python), [Microsoft sbom-tool](https://github.com/microsoft/sbom-tool).

## Commands

```powershell
# Image + project SBOMs (CycloneDX + SPDX JSON)
.\tasks.ps1 sbom

# Image only (OS packages + Python deps inside kepler-engine:local)
.\scripts\sbom.ps1 image
.\scripts\sbom.ps1 image -SkipBuild

# Project filesystem only (uv.lock / source tree)
.\scripts\sbom.ps1 project
```

## Outputs (gitignored — keep private)

| File | Contents |
|---|---|
| `reports/sbom/kepler-engine.cdx.json` | CycloneDX for the app container image |
| `reports/sbom/kepler-engine.spdx.json` | SPDX for the app container image |
| `reports/sbom/project.cdx.json` | CycloneDX for the repo / lockfile |
| `reports/sbom/project.spdx.json` | SPDX for the repo / lockfile |

## Scan an SBOM for vulnerabilities

```powershell
# Requires existing CycloneDX files under reports/sbom/
.\scripts\sbom.ps1 scan
.\scripts\sbom.ps1 scan -Severity MEDIUM
.\scripts\sbom.ps1 scan -FailOnFindings
```

Writes local (gitignored) reports:

| File | Contents |
|---|---|
| `reports/sbom/kepler-engine-scan.txt` / `.json` | Vulns from the image SBOM |
| `reports/sbom/project-scan.txt` / `.json` | Vulns from the project SBOM |

Same privacy rule: keep scan output off the public repo.
