#!/usr/bin/env bash

BASEPATH="/private/home/akadian"

module load anaconda3
source activate tfv10
module load cuda/9.0
module load cudnn/v7.3-cuda.9.2
export LD_LIBRARY_PATH="/public/apps/cuda/9.0/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}"
export PYTHONPATH="${BASEPATH}/esp-master/esp:${BASEPATH}/esp-master/esp/build/esp/bindings:${PYTHONPATH}"
unset DISPLAY

cd "${BASEPATH}/rl-openai-baselines"


pip install -e .
pytest
