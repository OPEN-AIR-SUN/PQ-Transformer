from ast import dump

import matplotlib.pyplot as plt

from models.dump_helper import dump_results
from models.dump_helper_quad import dump_results_quad
from fit import fit_gamma

import os
import sys
import time
import numpy as np
import json
import argparse
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from IPython import embed
from utils.lr_scheduler import get_scheduler
from utils.logger import setup_logger
from models.pq_transformer import PQ_Transformer
from models.loss_helper_pq import get_loss
from models.ap_helper_pq import APCalculator, parse_predictions, parse_groundtruths, QUADAPCalculator, parse_quad_predictions, parse_quad_groundtruths

from tqdm import tqdm

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


def initiate_environment(args):
    '''
    initiate randomness.
    :param config:
    :return:
    '''
    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    np.random.seed(args.rng_seed)
    random.seed(args.rng_seed)
    

def load_checkpoint(args, model, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(args.checkpoint_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    logger.info('==> Saving...')
    state = {
        'config': args,
        'save_path': '',
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }

    if save_cur:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    elif epoch % args.save_freq == 0:
        state['save_path'] = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth'))
        logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')))
    else:
        print("not saving checkpoint")
        pass


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
        TRAIN_DATASET = ScannetDetectionDataset('train', num_points=args.num_point,
                                                augment=True,
                                                use_color=True if args.use_color else False,
                                                use_height=True if args.use_height else False)

        TEST_DATASET = ScannetDetectionDataset('val', num_points=args.num_point,
                                               augment=False,
                                               use_color=True if args.use_color else False,
                                               use_height=True if args.use_height else False)
    else:
        raise NotImplementedError(f'Unknown dataset {args.dataset}. Exiting...')

    print(f"train_len: {len(TRAIN_DATASET)}, test_len: {len(TEST_DATASET)}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(TRAIN_DATASET)
    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               worker_init_fn=my_worker_init_fn,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)

    test_sampler = torch.utils.data.distributed.DistributedSampler(TEST_DATASET, shuffle=False)
    test_loader = torch.utils.data.DataLoader(TEST_DATASET,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              worker_init_fn=my_worker_init_fn,
                                              pin_memory=True,
                                              sampler=test_sampler,
                                              drop_last=False)
    print(f"train_loader_len: {len(train_loader)}, test_loader_len: {len(test_loader)}")

    return train_loader, test_loader, DATASET_CONFIG


def get_model(args, DATASET_CONFIG):
    if args.use_height:
        num_input_channel = int(args.use_color) * 3 + 1
    else:
        num_input_channel = int(args.use_color) * 3
    model = PQ_Transformer(num_class=DATASET_CONFIG.num_class,
                              num_heading_bin=DATASET_CONFIG.num_heading_bin,
                              num_size_cluster=DATASET_CONFIG.num_size_cluster,
                              mean_size_arr=DATASET_CONFIG.mean_size_arr,
                              input_feature_dim=num_input_channel,
                              num_proposal=args.num_target,
                              num_quad_proposal=args.quad_num_target,
                              sampling=args.sampling)
    criterion = get_loss
    return model, criterion


def main(args):
    train_loader, test_loader, DATASET_CONFIG = get_loader(args)
    n_data = len(train_loader.dataset)
    logger.info(f"length of training dataset: {n_data}")
    n_data = len(test_loader.dataset)
    logger.info(f"length of testing dataset: {n_data}")

    model, criterion = get_model(args, DATASET_CONFIG)
    if dist.get_rank() == 0:
        logger.info(str(model))
    # optimizer
    if args.optimizer == 'adamW':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "coder" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "coder" in n and p.requires_grad],
                "lr": args.decoder_learning_rate,
            },
        ]
        optimizer = optim.AdamW(param_dicts,
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    scheduler = get_scheduler(optimizer, len(train_loader), args)
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    if args.checkpoint_path:
        assert os.path.isfile(args.checkpoint_path)
        load_checkpoint(args, model, optimizer, scheduler)

    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
                   'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                   'per_class_proposal': True, 'conf_thresh': 0.0,
                   'dataset_config': DATASET_CONFIG}

    evaluate_one_epoch(test_loader, DATASET_CONFIG, CONFIG_DICT, args.ap_iou_thresholds, model, criterion, args)
    save_checkpoint(args, 'last', model, optimizer, scheduler, save_cur=True)
    logger.info("Saved in {}".format(os.path.join(args.log_dir, f'ckpt_epoch_last.pth')))
    return os.path.join(args.log_dir, f'ckpt_epoch_last.pth')



def evaluate_one_epoch(test_loader, DATASET_CONFIG, CONFIG_DICT, AP_IOU_THRESHOLDS, model, criterion, config):
    
    stat_dict = {}


    if config.num_decoder_layers > 0:
        prefixes = ['last_'] #, 'proposal_'] + [f'{i}head_' for i in range(config.num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
                          for iou_thresh in AP_IOU_THRESHOLDS]
    
    quad_ap_calculator_list = [QUADAPCalculator(iou_thresh, DATASET_CONFIG.class2quad) \
                          for iou_thresh in AP_IOU_THRESHOLDS]
    

    mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in AP_IOU_THRESHOLDS]

    model.eval()  # set model to eval mode (for bn and dp)
    batch_pred_map_cls_dict = {k: [] for k in prefixes}
    batch_gt_map_cls_dict = {k: [] for k in prefixes}

    batch_pred_quad_map_cls_dict = {k: [] for k in prefixes}
    batch_gt_quad_map_cls_dict = {k: [] for k in prefixes}

    batch_pred_corner_dict = {k: [] for k in prefixes}
    batch_gt_corner_dict = {k: [] for k in prefixes}

    batch_gt_horizontal_dict = {k: [] for k in prefixes}
    for batch_idx, batch_data_label in tqdm(enumerate(test_loader)):

        for key in batch_data_label:
            if key == 'scan_name':
                continue
            
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = model(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        

        loss, end_points = criterion(end_points, DATASET_CONFIG, pc_loss = False)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                if isinstance(end_points[key], float):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()

        for prefix in prefixes:
            batch_pred_map_cls, pred_mask = parse_predictions(end_points, CONFIG_DICT, prefix)
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)
            end_points['pred_mask']=pred_mask


            batch_pred_quad_map_cls,pred_quad_mask,batch_pred_quad_corner = parse_quad_predictions(end_points, CONFIG_DICT, prefix)
            batch_gt_quad_map_cls,batch_gt_quad_corner = parse_quad_groundtruths(end_points, CONFIG_DICT)
            
            batch_pred_quad_map_cls_dict[prefix].append(batch_pred_quad_map_cls)
            batch_gt_quad_map_cls_dict[prefix].append(batch_gt_quad_map_cls)
            batch_pred_corner_dict[prefix].append(batch_pred_quad_corner)
            batch_gt_corner_dict[prefix].append(batch_gt_quad_corner)
            
            batch_gt_horizontal_dict[prefix].append(end_points['horizontal_quads']) 

            end_points['pred_quad_mask']=pred_quad_mask

            # === Distance Loss ===
            for b in range(end_points['point_clouds'].shape[0]):
                pc_scene = end_points['point_clouds'][b]
                semantic_labels = end_points['semantic_labels'][b]

                predicted_quads = batch_pred_quad_corner[b]

                # Filter out all points inside predicted quads

                for predicted_quad in predicted_quads:
                    lower_bound = np.min(predicted_quad, axis=0)
                    upper_bound = np.max(predicted_quad, axis=0)

                    outside = np.zeros(shape=(pc_scene.shape[0], ))

                    for i in range(3):
                        outside[torch.logical_or(pc_scene[:, i] < lower_bound[i], upper_bound[i] < pc_scene[:, i]).cpu().numpy()] = 1

                    # pc_scene: (#num_points, 3) -> (#num_points_not_in_predicted_quad, 3)
                    # pc_scene = pc_scene[np.argwhere(outside > 0).reshape(-1), :]
                    # semantic_labels = semantic_labels[np.argwhere(outside > 0).reshape(-1)]



                pc_center = torch.mean(pc_scene, dim=0).cpu().numpy()  # To find the inner side of the point clouds

                # pc_wall_scene = pc_scene[semantic_labels == 1]
                mask = semantic_labels == 1
                mask = torch.bitwise_or(mask, semantic_labels == 8)
                mask = torch.bitwise_or(mask, semantic_labels == 9)
                distance = 10.0 + np.zeros(shape=(pc_scene.shape[0], ))
                # Calculate distances
                for predicted_quad in predicted_quads:
                    quad_center = np.mean(predicted_quad, axis=0)
                    predicted_quad_norm = np.cross(predicted_quad[1] - predicted_quad[0], predicted_quad[2] - predicted_quad[0])
                    predicted_quad_norm = predicted_quad_norm / np.linalg.norm(predicted_quad_norm)

                    if np.dot(pc_center - quad_center, predicted_quad_norm) > 0:
                        predicted_quad_norm = -predicted_quad_norm  # Make inner distance < 0 and outsider > 0

                    new_distance = np.dot(pc_scene.cpu().numpy() - quad_center, predicted_quad_norm)

                    distance[(np.abs(new_distance) < np.abs(distance))] = new_distance[(np.abs(new_distance) < np.abs(distance))]

                distance_wall = distance[mask.cpu().numpy()]
                embed()
                np.save(f'statistics_zcy/distance-{end_points["scan_idx"][b].item()}.npy', distance_wall)
                plt.cla()
                plt.hist(distance_wall, bins=100, color='g', alpha=0.5)
                plt.savefig(f'statistics_zcy/distribution-{end_points["scan_idx"][b].item()}.png')

                distance = np.abs(distance)
                distance_wall = distance[mask.cpu().numpy()]
                d2c = lambda d: ((128 + min(1, d) * 127)/255, (255-min(1, d) * 127)/255, 1.)
                with open(f'statistics_zcy/distance-{end_points["scan_idx"][b].item()}.txt', 'w') as f:
                    for i in range(len(pc_scene)):
                        if mask[i]:
                            color = d2c(distance[i])
                            # color = (255,255,255)
                        else:
                            color = (64/256,64/256,64/256)
                        print(f"{pc_scene[i][0]} {pc_scene[i][1]} {pc_scene[i][2]} {color[0]} {color[1]} {color[2]}", file=f)
                    
                np.save(f'statistics_zcy/distance-abs-{end_points["scan_idx"][b].item()}.npy', distance_wall)
                plt.cla()
                plt.hist(distance_wall, bins=100, color='g', alpha=0.5)
                plt.savefig(f'statistics_zcy/distribution-{end_points["scan_idx"][b].item()}-abs.png')

            
            if config.dump_result:
                print("dumping...")
                dump_results(end_points, os.path.join(ROOT_DIR,'dump/%01dbest'%(batch_idx)), DATASET_CONFIG) 
                dump_results_quad(end_points, os.path.join(ROOT_DIR,'dump/%01dbest'%(batch_idx)), DATASET_CONFIG) 
            

        if (batch_idx + 1) % config.print_freq == 0:
            logger.info(f'Eval: [{batch_idx + 1}/{len(test_loader)}]  ' + ''.join(
                [f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                 for key in sorted(stat_dict.keys()) if 'loss' not in key]))
            logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                 for key in sorted(stat_dict.keys()) if
                                 'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key]))
            logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                 for key in sorted(stat_dict.keys()) if 'last_' in key]))
            logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                 for key in sorted(stat_dict.keys()) if 'proposal_' in key]))
            for ihead in range(config.num_decoder_layers - 2, -1, -1):
                logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                                     for key in sorted(stat_dict.keys()) if f'{ihead}head_' in key]))

    #objects:
    mAP = 0.0
    for prefix in prefixes:
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
                                                          batch_gt_map_cls_dict[prefix]):
        
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        # Evaluate average precision
        
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            logger.info(f'=====================>{prefix} IOU THRESH: {AP_IOU_THRESHOLDS[i]}<=====================')
            for key in metrics_dict:
                logger.info(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

    for mAP in mAPs:
        logger.info(f'IoU[{mAP[0]}]:\t' + ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))

    object_map = mAP[1]['last_']
    #quad
    mAP_ = 0.0
    
    for prefix in prefixes:
        for (batch_pred_map_cls, batch_gt_map_cls,batch_pred_corner,batch_gt_corner,batch_gt_horizontal) in zip(batch_pred_quad_map_cls_dict[prefix],
                                                          batch_gt_quad_map_cls_dict[prefix],batch_pred_corner_dict[prefix],batch_gt_corner_dict[prefix],batch_gt_horizontal_dict[prefix]):
            for ap_calculator in quad_ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls,batch_pred_corner,batch_gt_corner,batch_gt_horizontal)
        # Evaluate average precision
        for i, ap_calculator in enumerate(quad_ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            f1 = ap_calculator.compute_F1(calculated=True)
            logger.info(f'F1 scores: {f1}')
            # logger.info(f'=====================>{prefix} IOU THRESH: {AP_IOU_THRESHOLDS[i]}<=====================')
            # for key in metrics_dict:
            #     logger.info(f'{key} {metrics_dict[key]}')
            if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                mAP_ = metrics_dict['mAP']
            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()

    for mAP_ in mAPs:
        logger.info(f'IoU[{mAP_[0]}]:\t' + ''.join([f'{key}: {mAP_[1][key]:.4f} \t' for key in sorted(mAP_[1].keys())]))
 

    return mAP

if __name__ == '__main__':
    opt = parse_option()
    
    torch.cuda.set_device(opt.local_rank if opt.local_rank else 0)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    initiate_environment(opt)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    LOG_DIR = os.path.join(opt.log_dir, 'pq-transformer',
                           f'{opt.dataset}_{int(time.time())}', f'{np.random.randint(100000000)}')
    while os.path.exists(LOG_DIR):
        LOG_DIR = os.path.join(opt.log_dir, 'pq-transformer',
                               f'{opt.dataset}_{int(time.time())}', f'{np.random.randint(100000000)}')
    opt.log_dir = LOG_DIR
    os.makedirs(opt.log_dir, exist_ok=True)

    logger = setup_logger(output=opt.log_dir, distributed_rank=dist.get_rank(), name="pq-transformer")
    if dist.get_rank() == 0:
        path = os.path.join(opt.log_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        logger.info(str(vars(opt)))

    ckpt_path = main(opt)
