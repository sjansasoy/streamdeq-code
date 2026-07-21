#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=4
GPUS_PER_NODE=4
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    -A romer \
    --job-name=${JOB_NAME} \
    --gres=gpu:4 \
    --ntasks=4 \
    --nodes 1 \
    --ntasks-per-node=4 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
