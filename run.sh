#!/usr/bin/env bash
# Single entrypoint for the whole pipeline. Run any single experiment by id —
# no need for run_all. All knobs (model, features, embeddings, HPO trials,
# sample weighting, seeds) live in configs/<exp_id>.yaml.
#
# Usage:
#   ./run.sh setup                  # CPU deps
#   ./run.sh setup-gpu              # + torch / sentence-transformers / transformers
#   ./run.sh folds                  # (re)create the fixed fold file if missing
#   ./run.sh exp005                 # run ONE experiment from configs/exp005.yaml
#   ./run.sh exp010 --hpo-trials 200   # CLI overrides the config's hpo.trials
#   ./run.sh all [--force]          # every config, skipping completed exp_ids
#   ./run.sh blend [exp ids...]     # blend all artifacts, or only the listed ones
#   ./run.sh submit exp005          # write submissions/sub_exp005.csv + sanity report
#   ./run.sh eda                    # regenerate reports/figures/
#   ./run.sh adv                    # adversarial validation
#
# Note: running an experiment again overwrites its artifacts and appends a new
# row to results.csv (the latest row per exp_id is the one that counts).
set -euo pipefail
cd "$(dirname "$0")"

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

cmd=${1:-}
[ -z "$cmd" ] && { usage; exit 1; }
shift || true

case "$cmd" in
  -h|--help|help) usage ;;
  setup)     pip install -r requirements.txt ;;
  setup-gpu) pip install -r requirements.txt torch sentence-transformers transformers ;;
  folds)     python -c "from src.data import make_folds; print(make_folds()['fold'].value_counts().sort_index())" ;;
  all)       python scripts/run_all.py "$@" ;;
  blend)     if [ $# -gt 0 ]; then python scripts/blend.py -e "$@"; else python scripts/blend.py; fi ;;
  submit)    python scripts/make_submission.py -e "${1:?usage: ./run.sh submit <exp_id|blend>}" ;;
  eda)       python reports/eda/deep_eda.py ;;
  adv)       python scripts/adversarial_validation.py ;;
  *)
    cfg="configs/$cmd.yaml"
    [ -f "$cfg" ] || { echo "unknown command or config: $cmd ($cfg not found)"; echo; usage; exit 1; }
    python scripts/run_experiment.py -c "$cfg" "$@"
    ;;
esac
