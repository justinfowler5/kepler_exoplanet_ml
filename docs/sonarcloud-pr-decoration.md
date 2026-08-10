# SonarCloud GitHub PR decoration — one-time setup

Repo: https://github.com/justinfowler5/kepler_exoplanet_ml
Workflow: .github/workflows/sonarcloud.yml
Project key: justinfowler5_kepler_exoplanet_ml
Organization: justinfowler5

## Why SonarCloud (not the local Docker SonarQube)

PR decoration needs a Sonar instance that can talk to GitHub and a **bound** GitHub project.
The local `kepler-sonarqube` container on localhost:9002 cannot decorate GitHub PRs.

## Steps (browser + one secret)

1. Sign in at https://sonarcloud.io with GitHub (or https://sonarqube.us for the US region).
2. Create/import organization for GitHub user `justinfowler5` and install the **SonarQube Cloud** GitHub App on `kepler_exoplanet_ml` (or the whole account).
3. Import / analyze `kepler_exoplanet_ml` so the SonarCloud project is **bound** to the GitHub repo.
   - Confirm the project key matches `justinfowler5_kepler_exoplanet_ml` (SonarCloud's default for this repo).
   - If SonarCloud assigned a different key, update `sonar.projectKey` / `sonar.organization` in `sonar-project.properties`.
4. Create a token: SonarCloud → My Account → Security → Generate Token.
5. Add the token as a GitHub Actions secret named `SONAR_TOKEN` on the repo:
   - GitHub → Settings → Secrets and variables → Actions → New repository secret
   - Name: `SONAR_TOKEN`
   - Value: (the token from step 4)
6. Commit and push `.github/workflows/sonarcloud.yml` + updated `sonar-project.properties` to `main`, then open a PR.

## What you get on each PR

- SonarCloud Check / Quality Gate status on the PR
- Inline issue annotations on changed lines
- Coverage from `pytest-cov` (`coverage.xml`) included in the analysis

## Optional: fail the GitHub job on Quality Gate failure

Already enabled via `-Dsonar.qualitygate.wait=true` in the workflow.

## Optional: require Quality Gate to merge

GitHub → Settings → Branches → Branch protection rule for `main`
→ Require status checks → enable the SonarCloud / Quality Gate check once it has run once.
