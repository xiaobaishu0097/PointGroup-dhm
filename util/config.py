'''
config.py
Written by Li Jiang
'''

import argparse
import yaml
import os
import torch

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/pointgroup_default_scannet.yaml', help='path to config file')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    ### distributed training
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args_cfg.rank = int(os.environ["RANK"])
        args_cfg.world_size = int(os.environ['WORLD_SIZE'])
        args_cfg.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args_cfg.rank = int(os.environ['SLURM_PROCID'])
        args_cfg.gpu = args_cfg.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args_cfg.distributed = False
        return args_cfg

    args_cfg.distributed = False
    # args_cfg.distributed = True
    #
    # torch.cuda.set_device(args_cfg.gpu)
    # args_cfg.dist_backend = 'nccl'
    # print('| distributed init (rank {}): {}'.format(
    #     args_cfg.rank, args_cfg.dist_url), flush=True)
    # torch.distributed.init_process_group(backend=args_cfg.dist_backend, init_method=args_cfg.dist_url,
    #                                      world_size=args_cfg.world_size, rank=args_cfg.rank)
    # torch.distributed.barrier()
    # setup_for_distributed(args_cfg.rank == 0)

    return args_cfg


cfg = get_parser()
setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
