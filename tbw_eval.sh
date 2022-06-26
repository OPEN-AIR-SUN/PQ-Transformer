#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export PORT=33333

python -m torch.distributed.launch --nproc_per_node 1 --master_port $PORT eval_tbw.py --checkpoint_path ./pretrained_model/ckpt_epoch_last.pth --pc_loss