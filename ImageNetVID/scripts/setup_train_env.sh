#!/usr/bin/env bash
# Sets up a conda environment for TRAINING StreamDEQ on ImageNet-VID: PyTorch
# 1.10.0 (the version the original authors used, confirmed by the README and
# by the official checkpoint's metadata) + mmcv-full 1.2.7 built from source
# (no prebuilt wheel of mmcv-full 1.2.7 exists for PyTorch >1.7.0, so it must
# be compiled locally).
#
# Usage:
#   bash scripts/setup_train_env.sh [env_name]
# (run from ImageNetVID/'s root, or the script will locate itself)
#
# Tested on Ubuntu 22.04. On a different base OS, steps 3-5 (CUDA toolkit,
# host compiler, crypt.h) are the most likely to need adjusting -- if the
# mmcv build fails at a different point, read the actual error message rather
# than assuming this script covers every case.

set -e  # stop on first error rather than continuing with a half-built env

ENV_NAME="${1:-streamdeq_vid_train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"   # ImageNetVID/

echo "=== [1/9] Creating conda env '$ENV_NAME' (Python 3.8) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$ENV_NAME" python=3.8 -y
conda activate "$ENV_NAME"

echo "=== [2/9] Installing PyTorch 1.10.0 (via conda, as the README instructs) ==="
conda install -y pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.1 -c pytorch -c conda-forge

echo "=== [3/9] Installing a CUDA toolkit WITH a compiler (nvcc) ==="
# PyTorch only requires the MAJOR CUDA version to match (11.x with 11.x), not
# an exact match -- so there's no need to hunt down 11.1 specifically. If the
# system's own nvcc is already 11.x, this step is redundant but harmless.
conda install -y -c conda-forge cudatoolkit-dev=11.4

echo "=== [4/9] Installing GCC/G++ 9 (nvcc 11.4 rejects newer host compilers) ==="
conda install -y -c conda-forge gxx_linux-64=9 gcc_linux-64=9

echo "=== [5/9] Preparing the crypt.h shim ==="
# On many modern distros crypt.h moved out of glibc (into libxcrypt), and
# conda-forge's isolated sysroot doesn't always include it -- Python.h needs
# it to compile. This builds a directory containing ONLY that header, rather
# than exposing all of /usr/include (which can conflict with conda's own
# sysroot headers).
CRYPT_SHIM="$SCRIPT_DIR/.crypt_shim"
mkdir -p "$CRYPT_SHIM"
if [ -f /usr/include/crypt.h ]; then
    ln -sf /usr/include/crypt.h "$CRYPT_SHIM/crypt.h"
else
    echo "  NOTE: /usr/include/crypt.h does not exist on this machine -- if the"
    echo "  mmcv/mmpycocotools build fails on crypt.h, install libxcrypt-dev via"
    echo "  apt, or 'conda install -c conda-forge libxcrypt', and retry."
fi

CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CC CXX
export CUDA_HOME="$CONDA_PREFIX"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export CPATH="$CRYPT_SHIM"

echo "=== [6/9] Building mmcv-full 1.2.7 from source ==="
MMCV_SRC="$SCRIPT_DIR/.mmcv_src"
rm -rf "$MMCV_SRC"
git clone --branch v1.2.7 --depth 1 https://github.com/open-mmlab/mmcv.git "$MMCV_SRC"
pip install -r "$MMCV_SRC/requirements.txt"
(cd "$MMCV_SRC" && pip install -e . --no-build-isolation)

echo "=== [7/9] Building mmpycocotools (requires Cython<3) ==="
pip install "cython<3"
COCOAPI_SRC="$SCRIPT_DIR/.cocoapi_src"
rm -rf "$COCOAPI_SRC"
git clone --depth 1 https://github.com/open-mmlab/cocoapi.git "$COCOAPI_SRC"
(cd "$COCOAPI_SRC/pycocotools" && pip install . --no-build-isolation)

echo "=== [8/9] Installing mmdet==2.10.0 (no deps) + this repo as editable ==="
pip install mmdet==2.10.0 --no-deps
(cd "$REPO_ROOT" && pip install -e . --no-deps --no-build-isolation)

echo "=== [9/9] Loose dependencies + compatible pins + missing lib/ ==="
# yacs/scipy/termcolor/terminaltables: not listed in requirements/runtime.txt
# but required at import time.
# numpy<1.24: this codebase uses np.int/np.bool, removed in numpy 1.24+.
# yapf<0.32: mmcv 1.2.7 calls a yapf API that newer yapf versions dropped.
pip install yacs scipy termcolor terminaltables "numpy==1.23.5" "yapf==0.31.0"

# mmdet/models/backbones/lib/ is gitignored in this repo and never committed
# to git -- restored here from the bundled copy.
mkdir -p "$REPO_ROOT/mmdet/models/backbones/lib"
cp "$SCRIPT_DIR/vendor_lib/"*.py "$REPO_ROOT/mmdet/models/backbones/lib/"

echo
echo "=== Final verification ==="
cd "$REPO_ROOT"
python -c "
import torch, mmcv, mmdet
print('torch', torch.__version__, '| mmcv', mmcv.__version__, '| mmdet', mmdet.__version__)
print('cuda available:', torch.cuda.is_available())
from mmdet.models.backbones.mdeq import MDEQNet
from mmdet.models.detectors.mdeq_faster_rcnn import MDEQ_FasterRCNN
print('Backbone/detector imports OK.')
"
echo
echo "Environment '$ENV_NAME' is ready. Activate it with: conda activate $ENV_NAME"
echo "To launch training: bash scripts/launch_training.sh <number_of_GPUs>"
