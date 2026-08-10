# Local Trivy container scanning

Uses the Aqua Trivy Docker image (no host install). Default target is the app image built from this repo's `Dockerfile`.

## Commands

```powershell
# Build kepler-engine:local and scan HIGH+CRITICAL (unfixed) vulns
.\tasks.ps1 trivy

# Same, with options
.\scripts\trivy.ps1 image
.\scripts\trivy.ps1 image -Severity MEDIUM
.\scripts\trivy.ps1 image -SkipBuild          # reuse existing tag
.\scripts\trivy.ps1 image -FailOnFindings     # non-zero exit if findings

# Also scan repo filesystem (vuln/secret/misconfig) and Dockerfile/compose config
.\scripts\trivy.ps1 all
.\scripts\trivy.ps1 fs
.\scripts\trivy.ps1 config
```

## Reports

Written under `reports/trivy/` (gitignored):

| File | Format |
|---|---|
| `image.txt` | Human-readable table |
| `image.json` | Machine-readable |
| `image.sarif` | SARIF (IDE / GitHub code scanning) |
| `fs.*` / `config.*` | From `fs` / `config` / `all` tasks |

## Notes

- Default image tag: `kepler-engine:local`
- Default severity floor: `HIGH` (also includes `CRITICAL`)
- `--ignore-unfixed` is on by default so noise from unpatched upstream CVEs is reduced
- Compose dependency images (`redis`, `postgres`, `minio`, `mlflow`) are not scanned by `image`; scan them explicitly if needed, e.g. `docker run --rm -v //var/run/docker.sock:/var/run/docker.sock aquasec/trivy:0.67.2 image redis:7-alpine`
