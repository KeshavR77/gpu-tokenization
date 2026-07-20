#!/usr/bin/env bash
# Creates the separate `slem_env` venv for the SLEM (HF UAD) arm.
# NEVER install any of this into the vLLM env -- vLLM nightly pins transformers.
set -euo pipefail

PY=${PYTHON:-python3}
DIR=${1:-slem_env}

$PY -m venv "$DIR"
"$DIR/bin/pip" install --upgrade pip
# transformers pin is load-bearing: slem_benchmark.py subclasses 5.14.1 internals
# and refuses to run against any other version.
"$DIR/bin/pip" install "transformers==5.14.1" "torch>=2.6" accelerate "huggingface_hub>=0.30" numpy

"$DIR/bin/python" - <<'EOF'
import torch, transformers
assert transformers.__version__ == "5.14.1", transformers.__version__
print("transformers", transformers.__version__)
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
EOF

echo
echo "slem_env ready. The draft repo (meta-llama/Llama-3.2-1B-Instruct) is GATED:"
echo "run '$DIR/bin/hf auth login' or export HF_TOKEN before running slem_benchmark.py."
echo "Smoke test: $DIR/bin/python slem_benchmark.py --k 4 --num-prompts 5"
