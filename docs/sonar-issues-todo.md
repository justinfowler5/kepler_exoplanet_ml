# SonarQube issues TODO

Captured from local SonarQube Community scan (project `kepler-engine`, Quality Gate OK).
Dashboard (if container is up): http://localhost:9002/dashboard?id=kepler-engine

**Snapshot metrics:** 0 bugs · 0 vulnerabilities · 0 security hotspots · **27 code smells** · **87% coverage** · 0% duplication

Re-scan after fixes: `.\scripts\sonar.ps1 all`

---

## Critical (4)

- [ ] `src/kepler_engine/services/ingestion.py:130` — Define a constant instead of duplicating literal `"ingestion.loaded"` 3 times (`python:S1192`)
- [ ] `src/kepler_engine/ml/evaluation.py:27` — Reduce cognitive complexity from 26 to ≤15 (`python:S3776`)
- [ ] `src/kepler_engine/services/inference_service.py:98` — Reduce cognitive complexity from 21 to ≤15 (`python:S3776`)
- [ ] `src/kepler_engine/ml/trainer.py:44` — Reduce cognitive complexity from 19 to ≤15 (`python:S3776`)

## Major (13)

- [ ] `src/kepler_engine/ml/preprocessing.py:31` — Remove unused function parameter `X` (`python:S1172`)
- [ ] `src/kepler_engine/ml/preprocessing.py:31` — Remove unused function parameter `y` (`python:S1172`)
- [ ] `src/kepler_engine/ml/preprocessing.py:55` — Remove unused function parameter `X` (`python:S1172`)
- [ ] `src/kepler_engine/ml/preprocessing.py:55` — Remove unused function parameter `y` (`python:S1172`)
- [ ] `src/kepler_engine/ml/preprocessing.py:66` — Remove unused function parameter `input_features` (`python:S1172`)
- [ ] `src/kepler_engine/ml/models.py:88` — Add missing hyperparameters `min_samples_leaf` and `max_features` for RandomForest (`python:S6973`)
- [ ] `src/kepler_engine/ml/models.py:88` — Provide a seed for `random_state` (`python:S6709`)
- [ ] `src/kepler_engine/ml/models.py:90` — Add missing hyperparameter `learning_rate` for GradientBoosting (`python:S6973`)
- [ ] `src/kepler_engine/ml/models.py:90` — Provide a seed for `random_state` (`python:S6709`)
- [ ] `tests/test_ingestion.py:132` — Exception test should have only one invocation that can throw (`python:S5778`)
- [ ] `tests/test_ingestion.py:140` — Exception test should have only one invocation that can throw (`python:S5778`)
- [ ] `tests/test_ingestion.py:148` — Exception test should have only one invocation that can throw (`python:S5778`)
- [ ] `tests/test_ingestion.py:164` — Exception test should have only one invocation that can throw (`python:S5778`)

## Minor (10)

- [ ] `src/kepler_engine/services/ingestion.py:64` — Remove redundant `Exception` in except clause (`python:S5713`) ×2 reported
- [ ] `src/kepler_engine/api/v1/health.py:12` — Remove redundant `response_model=` (duplicates return annotation) (`python:S8409`)
- [ ] `src/kepler_engine/api/v1/health.py:18` — Remove redundant `response_model=` (`python:S8409`)
- [ ] `src/kepler_engine/api/v1/experiments.py:21` — Remove redundant `response_model=` (`python:S8409`)
- [ ] `src/kepler_engine/api/v1/experiments.py:29` — Remove redundant `response_model=` (`python:S8409`)
- [ ] `src/kepler_engine/api/v1/experiments.py:46` — Remove redundant `response_model=` (`python:S8409`)
- [ ] `src/kepler_engine/api/v1/predictions.py:17` — Remove redundant `response_model=` (`python:S8409`)
- [ ] `src/kepler_engine/ml/trainer.py:83` — Specify a `memory` argument for the Pipeline (`python:S6969`)
- [ ] `src/kepler_engine/ml/preprocessing.py:108` — Specify a `memory` argument for the Pipeline (`python:S6969`)

---

# Trivy container issues TODO

Captured from local Aqua Trivy scan of `kepler-engine:local` (HIGH+CRITICAL, `--ignore-unfixed`).
Re-scan: `.\tasks.ps1 trivy` · Docs: `docs/trivy-local.md` · Reports: `reports/trivy/` (gitignored)

**Snapshot:** Debian OS layer clean · **1 HIGH** Python package finding · 0 CRITICAL

## High (1)

- [ ] `cryptography` **CVE-2026-69247** — installed `49.0.0`, fixed in `50.0.0` (transitive dep in image). Bump via lockfile / upstream constraint so the image rebuilds with ≥50.0.0. Ref: https://avd.aquasec.com/nvd/cve-2026-69247

---

# SBOM issues TODO

SBOMs generated with Trivy (`.\tasks.ps1 sbom`); vuln scan via `.\scripts\sbom.ps1 scan`.
Docs: `docs/sbom-local.md` · Artifacts: `reports/sbom/` (gitignored — keep private)

**Snapshot:** image SBOM ~219 components · project SBOM ~122 components · **1 HIGH** on both CycloneDX scans · 0 CRITICAL

## High (1) — confirmed via SBOM scan

- [ ] `cryptography` **CVE-2026-69247** — `49.0.0` → `50.0.0` (same finding on `kepler-engine.cdx.json` and `project.cdx.json`). After bumping, regenerate SBOMs and re-run scan:
  ```powershell
  .\tasks.ps1 sbom
  .\scripts\sbom.ps1 scan
  ```

## SBOM hygiene (optional)

- [ ] Keep regenerating SBOMs after dependency / image changes (`.\tasks.ps1 sbom`)
- [ ] Re-scan SBOMs after regenerating (`.\scripts\sbom.ps1 scan`)
- [ ] Do not commit `reports/sbom/` to the public repo

---

## Notes for later

- Several `preprocessing.py` “unused parameter” findings are likely sklearn transformer API hooks (`fit`/`transform` signatures). Prefer `_X` / `_y` naming or `# noqa` / Sonar exclusions over breaking the estimator contract.
- Cognitive-complexity items are refactors, not correctness bugs — tackle after Critical duplicate-literal and Major model seeding if time is limited.
- Related deferred work: SonarCloud PR decoration setup in `docs/sonarcloud-pr-decoration.md`.
- Trivy `fs` / `config` scans (`.\scripts\trivy.ps1 all`) are available but not yet snapshotted here.
