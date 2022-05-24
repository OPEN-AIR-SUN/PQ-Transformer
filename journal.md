[2022-5-22 19:45:17]
+ 安装依赖竟然失败了，scripy 提示找不到相应版本？后续如果影响使用再 Revise 一下这个问题
  + 这里是不是 typo? 应该装的是 scipy？
+ ~~尝试编译 pointnet2 失败，可能是 CUDA 版本不兼容？~~
  + 按照这个方式解决 https://github.com/facebookresearch/votenet/issues/108#issuecomment-783878066
+ 从 10.0.0.1 上直接偷来了训练数据
+ 下载了预训练模型，尝试运行
  + CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 eval.py --checkpoint_path ./pretrained_model/ckpt_epoch_last.pth
  + 预训练模型加载成功
+ 开始训模型（2022-5-22 20:35:53）
  + CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 train.py  --log_dir ./log --pc_loss
+ 训了 600 个 Epoch（2022-5-23 11:01:31）
  + 结果：/home/gaoha/PQ-Transformer/log/pq-transformer/scannet_1653274960/91571465/log.txt

[2022-5-24 15:16:47]
+ 开始读 Code Base
+ 