#!/bin/bash
# ============================================================
# setup_a100.sh
# Environment setup for biomedical SLM evaluation
# Pod: A100 PCIe | runpod/pytorch:2.4.0-py3.11-cuda12.4.1
# PyTorch upgraded 2.4.0 → 2.5.1 + vLLM 0.6.6
# Run once after pod starts: bash setup_a100.sh
# ============================================================

set -e

echo "============================================"
echo " Biomedical SLM Eval — A100 Setup Script"
echo " PyTorch 2.4.0 + CUDA 12.4 + Python 3.11"
echo "============================================"

# ------------------------------------------------------------
# 1. Environment variables
# ------------------------------------------------------------
echo "Setting environment variables..."

cat >> ~/.bashrc << 'EOF'
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache
export HF_HUB_DISABLE_XET=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0
EOF

export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache
export HF_HUB_DISABLE_XET=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0

# ------------------------------------------------------------
# 2. Create workspace directories
# ------------------------------------------------------------
echo "Creating workspace directories..."
mkdir -p /workspace/hf_cache
mkdir -p /workspace/models
mkdir -p /workspace/results

# ------------------------------------------------------------
# 3. Upgrade pip
# ------------------------------------------------------------
echo "Upgrading pip..."
pip install --upgrade pip -q

# ------------------------------------------------------------
# 4. Upgrade PyTorch to 2.5.1 + install matching vLLM 0.6.6
#    Template ships PyTorch 2.4.0 but vLLM 0.5.5 has broken
#    outlines dependencies. Upgrading to 2.5.1 + vLLM 0.6.6
#    is cleaner and fully supported on A100.
# ------------------------------------------------------------
echo "Upgrading PyTorch to 2.5.1 (CUDA 12.4)..."
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 -q

echo "Installing vLLM 0.6.6 (compatible with PyTorch 2.5.1)..."
pip uninstall vllm -y 2>/dev/null || true
pip install vllm==0.6.6 -q

# ------------------------------------------------------------
# 5. Transformers — pinned to avoid torchcodec/is_jax_tensor
# ------------------------------------------------------------
echo "Installing transformers (pinned)..."
pip install transformers==4.46.3 -q

# ------------------------------------------------------------
# 6. Evaluation metrics
# ------------------------------------------------------------
echo "Installing evaluation metrics..."
pip install rouge-score bert-score scikit-learn -q

# ------------------------------------------------------------
# 7. Data & utility libraries
# ------------------------------------------------------------
echo "Installing data & utility libraries..."
pip install datasets pandas "numpy<2.0" tqdm -q

# ------------------------------------------------------------
# 8. Dynamic few-shot — FAISS + sentence embeddings
#    Pin sentence-transformers to prevent upgrading transformers
# ------------------------------------------------------------
echo "Installing FAISS and sentence-transformers (pinned)..."
pip install "sentence-transformers<=2.7.0" faiss-cpu -q

# ------------------------------------------------------------
# 9. Fine-tuning / LoRA
# ------------------------------------------------------------
echo "Installing LoRA / fine-tuning libraries..."
pip install peft trl bitsandbytes accelerate -q

# ------------------------------------------------------------
# 10. Tokenizer & misc dependencies
# ------------------------------------------------------------
echo "Installing tokenizer dependencies..."
pip install sentencepiece protobuf -q
pip install "huggingface_hub<=0.25.0" -q
pip uninstall hf-xet -y 2>/dev/null || true   # remove Xet to force HTTP download

# ------------------------------------------------------------
# 11. Force reinstall transformers LAST
#     Must come after all other installs that may pull newer versions
# ------------------------------------------------------------
echo "Force-pinning transformers==4.46.3 (must be last)..."
pip install --force-reinstall transformers==4.46.3 -q

# ------------------------------------------------------------
# 12. Verify environment
# ------------------------------------------------------------
echo ""
echo "============================================"
echo " Verifying environment..."
echo "============================================"
nvidia-smi
python -c "
import torch
print(f'PyTorch  : {torch.__version__}')
print(f'CUDA     : {torch.version.cuda}')
print(f'GPU      : {torch.cuda.get_device_name(0)}')
print(f'VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'CUDA ok  : {torch.cuda.is_available()}')
import transformers
print(f'Transformers: {transformers.__version__}')
import vllm
print(f'vLLM     : {vllm.__version__}')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo " Next steps:"
echo "   1. huggingface-cli login"
echo "   2. Download model:"
echo "      huggingface-cli download <MODEL> \\"
echo "        --local-dir /workspace/models/<name> \\"
echo "        --local-dir-use-symlinks False"
echo "   3. Run your eval scripts"
echo "============================================"
