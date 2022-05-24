# TEST_dataloader_splitted
# To test if the splitted dataloader with the same random seed
# Would give the same result
import sys

import torch

sys.path.append("..")

from argparse import Namespace
from train import initiate_environment, get_loader, parse_option

if __name__ == "__main__":
    opt = parse_option()

    opt.end_proportion = 0.1

    result = [
        set(),
        set(),
        set()
    ]
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    for k in range(3):

        initiate_environment(opt)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        train_loader, test_loader, DATASET_CONFIG = get_loader(opt)

        for batch_idx, batch_data_label in enumerate(train_loader):
            for name in batch_data_label['scan_name']:
                result[k].add(name)

    # To test if three set in result are the same
    assert result[0] == result[1] == result[2], "\nFAILED"
    print("\nPASSED")



