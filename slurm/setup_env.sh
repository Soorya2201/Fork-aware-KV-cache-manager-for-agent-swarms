#!/bin/bash
# One-time environment setup. Run INSIDE a GPU allocation so the CUDA build matches:
#   salloc --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=1:00:00 --account=<pi>_123
#   bash slurm/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/carc.env"
cd "$PROJ"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install "sglang[all]"
uv pip install pandas numpy matplotlib scipy orjson transformers jinja2 openai python-dotenv requests httpx
uvx --from huggingface_hub huggingface-cli download "$MODEL"
python - <<'PY'
import torch, sglang
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("sglang", sglang.__version__)
PY
