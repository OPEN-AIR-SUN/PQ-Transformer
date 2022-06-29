export CUDA_VISIBLE_DEVICES=1 
export TORCH_DISTRIBUTED_DEBUG=DETAIL
python -m torch.distributed.launch --nproc_per_node 1 train_distance.py\
    --log_dir ./log \
    --pc_loss \
    --checkpoint_path /home/gaoha/PQ-Transformer/pretrained_model/ckpt_epoch_last.pth \
    --max_epoch 1200 \
