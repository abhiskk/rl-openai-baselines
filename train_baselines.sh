#!/usr/bin/env bash

BASEPATH="/private/home/akadian"
LOGFORMAT="stdout,log,csv,tensorboard"
LOGDIR="${BASEPATH}/logs/logs-rl-openai-baselines/pong/ppo2_iter_1"
NUM_ENV=32

module load anaconda3
source activate tfv10
module load cuda/9.0
module load cudnn/v7.3-cuda.9.2
export LD_LIBRARY_PATH="/public/apps/cuda/9.0/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia-opengl:${LD_LIBRARY_PATH}"
export PYTHONPATH="${BASEPATH}/esp-master/esp:${BASEPATH}/esp-master/esp/build/esp/bindings:${PYTHONPATH}"
unset DISPLAY

cd "${BASEPATH}/rl-openai-baselines"

export OPENAI_LOGDIR=${LOGDIR}
export OPENAI_LOG_FORMAT=${LOGFORMAT}

python -m baselines.run \
    --alg=ppo2 \
    --env=PongNoFrameskip-v4 \
    --num_timesteps 2e8 \
    --save_path "${BASEPATH}/models/pong_20M_ppo2" \
    --num_env ${NUM_ENV} \
