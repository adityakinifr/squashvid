#!/bin/bash
cd ~/code/squashvid; source .venv/bin/activate
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
for f in A B C; do python -I -m squashvid.ball.finetune fold $f --epochs 10 --patience 3 --batch 4 --workers 2 > work/ball_ft/fold$f.log 2>&1; done
