# Buildx bake definition for the Kepler app image (API + worker).
# Usage:  docker buildx bake -f docker-bake.hcl
#         .\scripts\build.ps1
#         .\tasks.ps1 build

variable "TAG" {
  default = "kepler-engine:local"
}

group "default" {
  targets = ["kepler"]
}

target "kepler" {
  context    = "."
  dockerfile = "Dockerfile"
  tags       = ["${TAG}"]
  # Load into the local Docker Engine (same as `docker buildx build --load`).
  # Uses the default docker driver on Desktop; no cache export (unsupported there).
  output = ["type=docker"]
  attest = [
    { type = "provenance", disabled = true },
    { type = "sbom", disabled = true },
  ]
}
