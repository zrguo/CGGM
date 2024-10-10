import random
import torch
import argparse
from src import foodtrain
from datasets.dataloader import getdataloader
import numpy as np


def foodrun():

    parser = argparse.ArgumentParser(description='Food')

    # Tasks
    parser.add_argument('--dataset', type=str, default='food')
    parser.add_argument('--modulation', type=str, default='cggm',
                        help='strategy to use (none/cggm)')
    parser.add_argument('--vit', type=str, default='',
                        help='pre-trained vit path for visual feature extraction')
    parser.add_argument('--bert', type=str, default='',
                        help='pre-trained bert path for textual feature extraction')
    parser.add_argument('--data_path', type=str, default='')

    # Dropouts
    parser.add_argument('--attn_dropout', type=float, default=0.2,
                        help='attention dropout')
    parser.add_argument('--relu_dropout', type=float, default=0.15,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.2,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.15,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.1,
                        help='output layer dropout')

    # Architecture
    parser.add_argument('--nlevels', type=int, default=4,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--cls_layers', type=int, default=2,
                        help='number of layers in the network (default: 2)')
    parser.add_argument('--num_heads', type=int, default=5,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--proj_dim', type=int, default=40,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')


    # Tuning
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--clip', type=float, default=0.8,
                        help='gradient clip value')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--cls_lr', type=float, default=5e-4,
                        help='classifier learning rate')
    parser.add_argument('--optim', type=str, default='AdamW',
                        help='optimizer to use')
    parser.add_argument('--num_epochs', type=int, default=60,
                        help='number of epochs')
    parser.add_argument('--when', type=int, default=10,
                        help='when to decay learning rate')
    parser.add_argument('--rou', type=float, default=1.3)
    parser.add_argument('--lamda', type=float, default=0.05)


    # Logistics
    parser.add_argument('--log_interval', type=int, default=30,
                        help='frequency of result logging')
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
    hyp_params.layers = args.nlevels
    hyp_params.use_cuda = use_cuda
    hyp_params.dataset = dataset
    hyp_params.when = args.when
    hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_loader), len(valid_loader), len(test_loader)
    hyp_params.output_dim = 101
    hyp_params.criterion = 'CrossEntropyLoss'
    hyp_params.num_mod = 2
    test_loss = foodtrain.initiate(hyp_params, train_loader, valid_loader, test_loader)


if __name__ == '__main__':
    foodrun()
