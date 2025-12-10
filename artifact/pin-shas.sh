#!/usr/bin/env bash
# Records git versions for the IntelliSA artifact.
# Prefers GitHub remotes when available; falls back to local HEAD when offline.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$SCRIPT_DIR/release-manifest.yaml"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PRIMARY_REPOS=("anon-hub" "anon-cli")
OPTIONAL_REPOS=("anon-experiments" "anon-models")

echo "Recording git versions (artifact focus: hub + cli; experiments/models optional)..."
echo ""

# Best-effort fetch from remotes (safe when offline)
echo "Fetching latest from GitHub (best-effort)..."
for repo in "${PRIMARY_REPOS[@]}" "${OPTIONAL_REPOS[@]}"; do
  if git -C "$ROOT/$repo" fetch origin main >/dev/null 2>&1; then
    echo "  $repo: fetch ok"
  else
    echo "  $repo: fetch skipped (offline or no remote)"
  fi
done
echo ""

remote_or_local() {
  local path="$1"
  git -C "$path" rev-parse origin/main 2>/dev/null || git -C "$path" rev-parse HEAD 2>/dev/null || echo "not-found"
}

HUB_SHA=$(remote_or_local "$ROOT/anon-hub")
CLI_SHA=$(remote_or_local "$ROOT/anon-cli")
EXPERIMENTS_SHA=$(remote_or_local "$ROOT/anon-experiments")
MODELS_SHA=$(remote_or_local "$ROOT/anon-models")
TIMESTAMP=$(date -u +"%Y-%m-%d")

echo "Recorded versions:"
echo "  hub:          $HUB_SHA"
echo "  cli:          $CLI_SHA"
echo "  experiments:  $EXPERIMENTS_SHA (optional)"
echo "  models:       $MODELS_SHA (optional)"
echo "  date:         $TIMESTAMP"
echo ""

echo "Checking for unpushed local changes..."
for repo in "${PRIMARY_REPOS[@]}" "${OPTIONAL_REPOS[@]}"; do
  cd "$ROOT/$repo"
  LOCAL=$(git rev-parse HEAD 2>/dev/null)
  REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "$LOCAL")
  if [ "$LOCAL" != "$REMOTE" ]; then
    echo "  ⚠️  $repo: Local differs from GitHub (unpushed commits?)"
  fi
done
echo ""

cat > "$MANIFEST" << EOF
# IntelliSA Paper Artifact - Version Record
# Records exact versions used to generate paper results

paper_title: "IntelliSA: An Intelligent Analyzer for IaC Security Smell Detection via Rule and Neural Inference"
paper_tag: v1.0-paper
generated_date: $TIMESTAMP

# Repository versions (recorded from GitHub remotes when available; otherwise local HEAD)
repositories:
  hub:
    github: https://github.com/ndryzsfdkx-source/anon-hub
    commit: $HUB_SHA

  cli:
    github: https://github.com/ndryzsfdkx-source/anon-cli
    commit: $CLI_SHA

  experiments:
    github: https://github.com/ndryzsfdkx-source/anon-experiments
    commit: $EXPERIMENTS_SHA
    optional: true

  models:
    github: https://github.com/ndryzsfdkx-source/anon-models
    commit: $MODELS_SHA
    optional: true

# Model used
model:
  name: codet5p-220m
  threshold: 0.61
  huggingface: https://huggingface.co/anonuser/anon-model-220m

# Dataset sizes
datasets:
  train: 1840
  validation: 230
  test: 230
  technologies: [ansible, chef, puppet]
EOF

echo "Updated: $MANIFEST"
echo "These are the versions currently recorded."
