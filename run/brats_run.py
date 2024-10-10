import random
import torch
import argparse
from src import segtrain
from datasets.dataloader import getdataloader
import numpy as np


def segrun():

    parser = argparse.ArgumentParser(description='Multimodal Segmentation')

    # Tasks
    parser.add_argument('--dataset', type=str, default='brats')
    parser.add_argument('--modulation', type=str, default='cggm',
                        help='strategy to use (none/cggm)')
    parser.add_argument('--data_path', type=str, default='')

    # Dropouts
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--start_warmup_value', type=float, default=4e-4)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--final_lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64, metavar='N')
    parser.add_argument('--clip', type=float, default=0.8)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cls_lr', type=float, default=6e-3)
    parser.add_argument('--num_epochs', type=int, default=75)
    parser.add_argument('--when', type=int, default=10)
    parser.add_argument('--rou', type=float, default=1.0)
    parser.add_argument('--lamda', type=float, default=0.1)


    # Logistics
    parser.add_argument('--log_interval', type=int, default=30,
                        help='frequency of result logging (default: 30)')
    parser.add_argument('--seed', type=int, default=666,
                        help='random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='do not use cuda')
    args = parser.parse_args()


    dataset = str.lower(args.dataset.strip())

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        if args.no_cuda:
            print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        else:
            use_cuda = True
    else:
        print('cuda not available!')

    setup_seed(args.seed)



    dataloder, orig_dim = getdataloader(args.dataset, args.batch_size, args.data_path)
    train_loader = dataloder['train']
    valid_loader = dataloder['valid']
    test_loader = dataloder['test']
    hyp_params = args
    hyp_params.orig_dim = orig_dim
    hyp_params.use_cuda = use_cuda
    hyp_params.dataset = dataset
    hyp_params.when = args.when
    hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_loader), len(valid_loader), len(test_loader)
    hyp_params.num_mod = 4

    test_loss = segtrain.initiate(hyp_params, train_loader, valid_loader, test_loader)


if __name__ == '__main__':
    segrun()
