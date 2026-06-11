#!/usr/bin/env bash
set -euo pipefail

G1_HOST="${G1_HOST:-g1-hotspot}"

rsync -avr --delete \
  --filter='P .venv*/' \
  --filter='P *.plan' \
  --filter='P *.engine' \
  --exclude='outputs/' \
  --exclude='outputs-orin/' \
  --exclude='outputs_play/' \
  --exclude='wandb/' \
  --exclude='memmap_td/' \
  --exclude='.git/' \
  --exclude='.cache' \
  --exclude='.codex' \
  --exclude='.omx' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.mypy_cache' \
  --exclude='stubs' \
  --exclude="checkpoints/lafan-old/" \
  --exclude="checkpoints/lafan/" \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --exclude='.DS_Store' \
  --exclude='sync*.sh' \
  --exclude='.venv' \
  --exclude='.venv*/' \
  --exclude='uv.lock' \
  --exclude='*.plan' \
  --exclude='*.engine' \
  --exclude='/datasets' \
  --exclude='sim2real/teleop/GMR/' \
  --exclude='mjcf/' \
  --exclude='external/' \
  --exclude='docs/' \
  --exclude='__pycache__/' \
  --exclude='*.nsys-rep' \
  --exclude='real_vr.tar' \
  --exclude='robot_motion_pair.npz' \
  /home/elijah/Documents/projects/simple-tracking/sim2real/ \
  "${G1_HOST}:/home/elijah/sim2real/"
  # g1-rp:/home/elijah/sim2real/
  # g1-xu:/home/unitree/haoyang/sim2real
  # g1-gao:/home/elijah/sim2real/
  # cl:/home/ubuntu/Desktop/haoyang/sim2real \
  # g1-gao:/home/unitree/haoyang/sim2real \

# rsync -avr \
#   --exclude='output_motion/' \
#   --exclude='LAFAN1_Retargeting_Dataset/' \
#   --exclude='.venv' \
#   --exclude='stubs' \
#   /home/elijah/Documents/projects/simple-tracking/lafan-process/ \
#   g1-rp:/home/elijah/lafan-process/ \

rsync -avr \
  --exclude='.git/' \
  --exclude='.venv' \
  --exclude='.venv*/' \
  --exclude='stubs' \
  --exclude='.cache' \
  --exclude='.codex' \
  --exclude='.omx' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.mypy_cache' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --exclude='.DS_Store' \
  --include='output/' \
  --include='output/cartwheel/***' \
  --include='output/lafan/' \
  --include='output/lafan/**' \
  --include='output/sonic/' \
  --include='output/sonic/manifest.json' \
  --include='output/sonic/motions/' \
  --include='output/sonic/motions/240529/' \
  --include='output/sonic/motions/240529/macarena_001__A545.npz' \
  --include='output/sonic/motions/230509/' \
  --include='output/sonic/motions/230509/forward_lunge_R_002__A359.npz' \
  --include='output/sonic/motions/230509/squat_001__A359.npz' \
  --include='output/sonic/motions/220713/' \
  --include='output/sonic/motions/220713/walk_backward_start_001__A021.npz' \
  --include='output/sonic/motions/240327/' \
  --include='output/sonic/motions/240327/one_leg_idle_R_002__A533.npz' \
  --exclude='output/sonic/**' \
  --exclude='output/sonic*' \
  --exclude='output/**' \
  /home/elijah/Documents/projects/simple-tracking/any4hdmi/ \
  "${G1_HOST}:/home/elijah/any4hdmi/"
  # g1-rp:/home/elijah/any4hdmi/
  # g1-xu:/home/unitree/haoyang/any4hdmi/
