'''
PointGroup train.py
Written by Li Jiang
'''

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import time, sys, os, random, glob
from tensorboardX import SummaryWriter
import numpy as np

import util.utils as utils

def init():
    # config
    global cfg
    from util.config import get_parser
    cfg = get_parser()
    cfg.dist = False

    # copy important files to backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    # logger
    global logger
    from util.log import get_logger
    logger = get_logger(cfg)

    # log the config
    logger.info(cfg)

    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)

def train_epoch(train_loader, model, model_fn, optimizer, epoch, scene_id):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()
    start_epoch = time.time()
    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        ##### adjust learning rate
        utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)

        ##### prepare input and forward
        loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch)

        ##### meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])

        ##### backward
        optimizer.zero_grad()
        loss.backward()
        if cfg.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        optimizer.step()

        ##### time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        sys.stdout.write(
            "epoch: {}/{} iter: {}/{} loss: {:.6f}({:.6f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
            (epoch, cfg.instance_classifier['epochs_per_instance'], i + 1, len(train_loader), am_dict['loss'].val, am_dict['loss'].avg,
             data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
        if (i == len(train_loader) - 1): print()


    logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.instance_classifier['epochs_per_instance'], am_dict['loss'].avg, time.time() - start_epoch))

    f = utils.checkpoint_scene_save(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, scene_id, cfg.save_freq)
    logger.info('Saving {}'.format(f))

    for k in am_dict.keys():
        if k in visual_dict.keys():
            writer.add_scalar(k+'_train', am_dict[k].avg, epoch)


def eval_epoch(val_loader, model, model_fn, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        model.eval()
        start_epoch = time.time()
        for i, batch in enumerate(val_loader):

            ##### prepare input and forward
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch)

            ##### meter_dict
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                am_dict[k].update(v[0], v[1])

            ##### print
            sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(i + 1, len(val_loader), am_dict['loss'].val, am_dict['loss'].avg))
            if (i == len(val_loader) - 1): print()

        logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)


if __name__ == '__main__':
    ##### init
    init()

    ##### SA
    if cfg.cache:
        if cfg.dataset == 'scannetv2':
            train_file_names = sorted(
                glob.glob(os.path.join(cfg.data_root, cfg.dataset, 'train', '*' + cfg.filename_suffix)))
            val_file_names = sorted(
                glob.glob(os.path.join(cfg.data_root, cfg.dataset, 'val', '*' + cfg.filename_suffix)))
            utils.create_shared_memory(train_file_names, wlabel=True)
            utils.create_shared_memory(val_file_names, wlabel=True)

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### dataset
    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            from data.scannetv2_inst import ScannetDatast
            dataset = ScannetDatast(cfg)
        else:
            print("Error: no data loader - " + data_name)
            exit(0)

        dataset.trainLoader()
        logger.info('Training samples: {}'.format(len(dataset.train_file_names)))
        dataset.valLoader()
        logger.info('Validation samples: {}'.format(len(dataset.val_file_names)))

    scene_id_restored = 0
    for scene_id in range(0, len(dataset.train_file_names)):
        if scene_id_restored > scene_id:
            continue
        # modify the instance number in the config file
        _, _, _, instance_label = dataset.train_files[scene_id]
        cfg.instance_classifier['instance_num'] = int(instance_label.max() + 1)

        dataset.trainInstanceLoader(scene_id)

        ##### model
        logger.info('=> creating model ...')

        if model_name == 'pointgroup':
            from model.pointgroup import PointGroup as Network
            from model.model_functions import model_fn_decorator
        else:
            print("Error: no model - " + model_name)
            exit(0)

        model = Network(cfg)

        use_cuda = torch.cuda.is_available()
        logger.info('cuda available: {}'.format(use_cuda))
        assert use_cuda
        model = model.cuda()

        # logger.info(model)
        logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

        ##### optimizer
        if cfg.optim == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
        elif cfg.optim == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum,
                                  weight_decay=cfg.weight_decay)

        ##### model_fn (criterion)
        model_fn = model_fn_decorator(cfg)

        ##### resume
        start_epoch, scene_id_restored, f = utils.checkpoint_scene_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5])      # resume from the latest epoch, or specify the epoch to restore
        if start_epoch > 10:
            start_epoch = 1
        if scene_id_restored > scene_id:
            continue
        logger.info('Restore from {}'.format(f) if len(f) > 0 else 'Start from epoch {}'.format(start_epoch))

        ##### train and val
        for epoch in range(start_epoch, cfg.instance_classifier['epochs_per_instance'] + 1):
            train_epoch(dataset.train_instance_data_loader, model, model_fn, optimizer, epoch, scene_id)

            # if utils.is_multiple(epoch, cfg.save_freq) or utils.is_power2(epoch):
            #     eval_epoch(dataset.val_data_loader, model, model_fn, epoch)

    ##### delete SA
    # if cfg.cache:
    #     if cfg.dataset == 'scannetv2':
    #         utils.delete_shared_memory(train_file_names, wlabel=True)
    #         utils.delete_shared_memory(val_file_names, wlabel=True)