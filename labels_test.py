# %%
import numpy as np
import sys
import os
import torch
import argparse
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
def parse_option():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_target', type=int, default=256, help='Object proposal number [default: 256]')
    parser.add_argument('--quad_num_target', type=int, default=256, help='Quad proposal number [default: 256]')
    parser.add_argument('--sampling', default='vote', type=str, help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--nhead', default=8, type=int, help='multi-head number')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dim_feedforward')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='transformer_dropout')
    parser.add_argument('--transformer_activation', default='relu', type=str, help='transformer_activation')

    # Data
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
    parser.add_argument('--dataset', default='scannet', help='Dataset name.  [default: scannet]')
    parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 50000]')
    parser.add_argument('--use_height', action='store_true', help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')

    # Training
    parser.add_argument('--optimizer', type=str, default='adamW', help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Optimization L2 weight decay [default: 0.0005]')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='Initial learning rate for all except decoder [default: 0.004]')
    parser.add_argument('--decoder_learning_rate', type=float, default=0.0001,
                        help='Initial learning rate for decoder [default: 0.0004]')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[320, 380], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='Default bn momeuntum')
    parser.add_argument('--syncbn', action='store_true', help='whether to use sync bn')

    # io
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--val_freq', type=int, default=10, help='val frequency')

    # others
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25], nargs='+',  #0.5
                        help='A list of AP IoU thresholds [default: 0.25,0.5]')
    parser.add_argument("--rng_seed", type=int, default=10, help='manual seed')
    parser.add_argument("--dump_result", action='store_true', help='dump')


    args, unparsed = parser.parse_known_args()

    return args

def get_loader(args):
    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Create Dataset and Dataloader
    if args.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet.scannet_detection_dataset import ScannetDetectionDataset
        from scannet.model_util_scannet import ScannetDatasetConfig

        DATASET_CONFIG = ScannetDatasetConfig()

        TEST_DATASET = ScannetDetectionDataset('val', num_points=args.num_point,
                                               augment=False,
                                               use_color=True if args.use_color else False,
                                               use_height=True if args.use_height else False)
    else:
        raise NotImplementedError(f'Unknown dataset {args.dataset}. Exiting...')

    test_loader = torch.utils.data.DataLoader(TEST_DATASET,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              worker_init_fn=my_worker_init_fn,
                                              pin_memory=True,
                                              drop_last=False)

    return None, test_loader, DATASET_CONFIG

args = parse_option()
_, test_loader, _ = get_loader(args)

# %%
a = next(iter(test_loader))

def save(points, labels, cls, name):
    with open(name, 'w') as f:
        for i in range(len(points)):
            if labels[i] == cls:
                print(f"{points[i][0]} {points[i][1]} {points[i][2]} 0.8 0.8 0.8", file=f)
            else:
                print(f"{points[i][0]} {points[i][1]} {points[i][2]} 0.2 0.2 0.2", file=f)

for i in range(41):
    save(a['point_clouds'][0], a['semantic_labels'][0], i, f'test_label_{i}.txt')

# %%
