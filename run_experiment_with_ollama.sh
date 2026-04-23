#!/usr/bin/env bash
set -euo pipefail

# Environment setup requested by user.
module load conda
conda activate /home/bma/conda_envs/chatbox
#conda init
#conda activate chatbox
#cd ChatBox_UniCA

# Ensure we are in repo root even if launched from elsewhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prerequisite commands requested by user.
echo 'export PATH=$PATH:~/ollama/bin' >> ~/.bashrc
# Sourcing the user's ~/.bashrc can fail when this script runs with "set -u"
# (nounset) if the .bashrc references variables that are not defined in this
# non-interactive context. Temporarily disable nounset while sourcing, then
# restore it. We also ignore source errors to avoid aborting the script.
set +u
if [ -f "$HOME/.bashrc" ]; then
	# shellcheck disable=SC1090
	source "$HOME/.bashrc" || true
fi
set -u

# ========== Force ollama to use only one GPU (GPU 0) ==========
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_NUM_GPU=1
# ================================================================

mkdir -p ~/ollama
nohup ~/ollama/bin/ollama serve > ~/ollama/ollama.log 2>&1 &

# Quick checks requested by user.
ollama list
ps aux | grep ollama || true
ollama pull gemma3:1b

# Launch experiment script with default settings for master.md only.
# Extra CLI args can still be appended to override these defaults.
python experiments/run_format_experiment.py \
  --auto-start-ollama \
  --docs docs/chroma/master.md \
  --question-file questions/questions_batch_student_short_typo_en_fr.txt \
  --gold-file questions/questions_batch_student_short_typo_en_fr_gold.json \
  --dynamic-topk-ratio 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  "$@"
