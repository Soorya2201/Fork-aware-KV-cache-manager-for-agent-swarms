#!/bin/bash
# Environment setup (called by setup.sbatch inside a GPU job so the CUDA build matches).
set -euo pipefail
source "$(dirname "$0")/carc.env"
cd "$PROJ"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv   # reruns reuse the venv
source .venv/bin/activate
uv pip install "sglang[all]"
uv pip install pandas numpy matplotlib scipy orjson transformers jinja2 openai python-dotenv requests httpx
# what the agent app itself imports (NOT its requirements.txt: those pin vllm 0.6.6 / torch 2.5
# for V100 and would break SGLang)
uv pip install python-docx huggingface_hub
# huggingface-cli was removed from huggingface_hub; the Python API works in every version
python -c "from huggingface_hub import snapshot_download as d; print(d('$MODEL'))"
python - <<'PY'
import torch, sglang
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("sglang", sglang.__version__)
PY
