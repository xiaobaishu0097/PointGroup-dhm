'''
PointGroup test.py
Written by Li Jiang
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import random
import os

from util.config import cfg
cfg.task = 'test'
from util.log import logger
import util.utils as utils
import util.eval as eval

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH), cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp test.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx
    semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39, -100]

    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def test(model, model_fn, data_name, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            from data.scannetv2_inst import ScannetDatast
            dataset = ScannetDatast(test=True)
            dataset.testLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)

    sampler_test = torch.utils.data.SequentialSampler(dataset.test_set)

    data_loader_test = DataLoader(
        dataset.test_set, batch_size=1, shuffle=False, sampler=sampler_test, collate_fn=dataset.testMerge,
        drop_last=False, num_workers=cfg.test_workers
    )

    maxpool3d = nn.MaxPool3d(3, stride=1, padding=1)

    with torch.no_grad():
        model = model.eval()
        start = time.time()

        true_threshold = 0.0
        candidate_num = 100

        matches = {}
        point_evaluations = {}

        for i, batch in enumerate(data_loader_test):
            N = batch['point_feats'].shape[0]
            test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1][:12]

            start1 = time.time()
            preds = model_fn(batch, model, epoch)
            end1 = time.time() - start1

            if not cfg.model_mode.endswith('_PointGroup'):
                grid_xyz = batch['grid_xyz'].cuda()

                ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)

                pt_offsets = preds['pt_offsets']    # (N, 3), float32, cuda
                pt_coords = preds['pt_coords']

                semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
                semantic_pred = semantic_scores[-1].max(1)[1]  # (N) long, cuda

                pt_semantic_labels = preds['pt_semantic_labels']
                pt_offset_labels = preds['pt_offset_labels']

                pt_valid_indx = (pt_semantic_labels > 1)

                if 'pt_semantic_eval' not in point_evaluations:
                    point_evaluations['pt_semantic_eval'] = {
                        'True': 0,
                        'False': 0,
                    }
                point_evaluations['pt_semantic_eval']['True'] += (
                        semantic_pred[pt_valid_indx] == pt_semantic_labels[pt_valid_indx]
                ).sum()
                point_evaluations['pt_semantic_eval']['False'] += (
                        semantic_pred[pt_valid_indx] != pt_semantic_labels[pt_valid_indx]
                ).sum()

                if 'pt_offset_eval' not in point_evaluations:
                    point_evaluations['pt_offset_eval'] = []
                point_evaluations['pt_offset_eval'].append(
                    (
                        torch.abs(
                            pt_offsets[-1][pt_valid_indx] - pt_offset_labels[pt_valid_indx]).sum() / pt_valid_indx.sum()
                    ).cpu().numpy()
                )

                # semantic_pred = semantic_scores  # (N) long, cuda
                # valid_index = (semantic_pred != 20)
                valid_index = (semantic_pred > 1)

                # semantic_pred = semantic_scores
                # semantic_pred[semantic_pred == -100] = 20
                # valid_index = (semantic_pred != 20)
                # semantic_pred[~valid_index] = 0

                grid_center_preds = torch.sigmoid(preds['centre_preds'])
                grid_center_semantic_preds = preds['centre_semantic_preds']
                grid_center_offset_preds = preds['centre_offset_preds']

                grid_pred_max = maxpool3d(grid_center_preds.reshape(1, 1, 32, 32, 32)).reshape(1, 32**3)
                cent_candidates_indexs = (grid_center_preds == grid_pred_max)
                grid_center_preds[~cent_candidates_indexs] = 0
                topk_value_, topk_index_ = torch.topk(grid_center_preds, candidate_num, dim=1)
                topk_index_ = topk_index_[topk_value_ > true_threshold]

                grid_xyz = grid_xyz.unsqueeze(dim=0)
                inst_cent_cand_xyz = grid_xyz[:, topk_index_, :] + grid_center_offset_preds[:, topk_index_, :]
                pt_cent_xyz = (pt_coords + pt_offsets[-1]).unsqueeze(dim=1)
                # pt_cent_xyz = pt_coords.permute(1, 0, 2)
                # l2 distance
                pt_inst_cent_dist = torch.norm(
                    pt_cent_xyz.repeat(1, inst_cent_cand_xyz.shape[1], 1) - inst_cent_cand_xyz.repeat(pt_cent_xyz.shape[0], 1, 1),
                    dim=2
                )
                # l1 distance
                # pt_inst_cent_dist = torch.sum(torch.abs(
                #     pt_cent_xyz.repeat(1, inst_cent_cand_xyz.shape[1], 1) - inst_cent_cand_xyz.repeat(pt_cent_xyz.shape[0], 1, 1)
                # ), dim=2)
                ### set the category restristion during instance generation
                inst_identity_mat = torch.nn.functional.one_hot(
                    grid_center_semantic_preds[topk_index_].to(torch.long), num_classes=cfg.classes+1
                ).float()
                pt_identity_mat = torch.nn.functional.one_hot(semantic_pred.to(torch.long), num_classes=cfg.classes+1).float()
                pt_inst_cov_mat = torch.mm(pt_identity_mat, inst_identity_mat.t())
                pt_inst_cov_mat[pt_inst_cov_mat == 0] = 1000
                pt_inst_cent_dist = pt_inst_cent_dist * pt_inst_cov_mat
                ### set the minimum distance threshold between point and grid point
                valid_dist_thresh = pt_inst_cent_dist.min(dim=1)[0] > cfg.TEST_DIST_THRESH
                pt_inst_cent = torch.nn.functional.one_hot(pt_inst_cent_dist.min(dim=1)[1], num_classes=topk_index_.shape[0])
                pt_inst_cent[valid_dist_thresh, :] = 0
                ### set the minimum point number threshold for each center
                valid_inst_index = pt_inst_cent.sum(dim=0) > cfg.TEST_NPOINT_THRESH
                pt_inst_cent = pt_inst_cent[:, valid_inst_index].permute(1, 0)
                pt_inst_cent[:, ~valid_index] = 0

                ### instance semantic label
                # TODO: decide which way to get instance semantic id
                ### 1. based on the majority point semantic labels in the instance
                # inst_semantic_id = torch.mm(pt_inst_cent.float(), torch.nn.functional.one_hot(semantic_pred.to(torch.int64)).float())
                # inst_semantic_id = inst_semantic_id.max(dim=1)[1]
                ### 2. based on grid center semantic predictions
                inst_semantic_id = grid_center_semantic_preds[topk_index_]
                inst_semantic_id = inst_semantic_id[valid_inst_index].long()

                inst_semantic_id = torch.tensor([semantic_label_idx[i] for i in inst_semantic_id])

                inst_scores = torch.ones((pt_inst_cent.shape[0]))

                ninst = pt_inst_cent.shape[0]

                ##### prepare for evaluation
                if cfg.eval:
                    pred_info = {}
                    pred_info['conf'] = inst_scores.cpu().numpy()
                    pred_info['label_id'] = inst_semantic_id.cpu().numpy()
                    pred_info['mask'] = pt_inst_cent.cpu().numpy()
                    gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)
                    matches[test_scene_name] = {}
                    matches[test_scene_name]['gt'] = gt2pred
                    matches[test_scene_name]['pred'] = pred2gt

            elif cfg.model_mode.endswith('_PointGroup'):
                ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
                if preds['semantic'][-1].shape[0] == N:
                    semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
                    semantic_pred = semantic_scores[-1].max(1)[1]  # (N) long, cuda
                else:
                    semantic_pred = preds['point_semantic_pred_full']
                # semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda

                pt_offsets = preds['pt_offsets']  # (N, 3), float32, cuda

                pt_semantic_labels = preds['pt_semantic_labels']
                pt_offset_labels = preds['pt_offset_labels']

                # stuff_preds = preds['stuff_preds'].max(1)[1]
                # stuff_labels = torch.zeros(stuff_preds.shape[0]).long().cuda()
                # stuff_labels[pt_semantic_labels > 1] = 1
                # pt_valid_indx = (stuff_labels == 1)
                pt_valid_indx = (semantic_pred > 1)

                if 'pt_semantic_eval' not in point_evaluations:
                    point_evaluations['pt_semantic_eval'] = {
                        'True': 0,
                        'False': 0,
                    }
                point_evaluations['pt_semantic_eval']['True'] += (
                        semantic_pred[pt_valid_indx] == pt_semantic_labels[pt_valid_indx]
                ).sum()
                point_evaluations['pt_semantic_eval']['False'] += (
                        semantic_pred[pt_valid_indx] != pt_semantic_labels[pt_valid_indx]
                ).sum()
                # point_evaluations['pt_semantic_eval']['True'] += (
                #     stuff_preds[pt_valid_indx] == stuff_labels[pt_valid_indx]
                # ).sum()
                # point_evaluations['pt_semantic_eval']['False'] += (
                #     stuff_preds[pt_valid_indx] != stuff_labels[pt_valid_indx]
                # ).sum()

                if preds['semantic'][-1].shape[0] == N:

                    if 'pt_offset_eval' not in point_evaluations:
                        point_evaluations['pt_offset_eval'] = []
                    point_evaluations['pt_offset_eval'].append(
                        (
                            torch.abs(pt_offsets[-1][pt_valid_indx] - pt_offset_labels[pt_valid_indx]).sum() / pt_valid_indx.sum()
                        ).cpu().numpy()
                    )

                if (epoch == cfg.test_epoch):
                    scores = preds['score']  # (nProposal, 1) float, cuda
                    scores_pred = torch.sigmoid(scores.view(-1))

                    proposals_idx, proposals_offset = preds['proposals']
                    # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                    # proposals_offset: (nProposal + 1), int, cpu
                    proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int,
                                                 device=scores_pred.device)  # (nProposal, N), int, cuda
                    proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

                    semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device)[
                        semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]]  # (nProposal), long

                    ##### score threshold
                    score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                    scores_pred = scores_pred[score_mask]
                    proposals_pred = proposals_pred[score_mask]
                    semantic_id = semantic_id[score_mask]

                    ##### npoint threshold
                    proposals_pointnum = proposals_pred.sum(1)
                    npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
                    scores_pred = scores_pred[npoint_mask]
                    proposals_pred = proposals_pred[npoint_mask]
                    semantic_id = semantic_id[npoint_mask]

                    ##### nms
                    if semantic_id.shape[0] == 0:
                        pick_idxs = np.empty(0)
                    else:
                        proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                        intersection = torch.mm(proposals_pred_f,
                                                proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                        proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                        proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                        proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                        cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                        pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(),
                                                        cfg.TEST_NMS_THRESH)  # int, (nCluster, N)
                    clusters = proposals_pred[pick_idxs]
                    cluster_scores = scores_pred[pick_idxs]
                    cluster_semantic_id = semantic_id[pick_idxs]

                    nclusters = clusters.shape[0]

                    ##### prepare for evaluation
                    if cfg.eval:
                        pred_info = {}
                        pred_info['conf'] = cluster_scores.cpu().numpy()
                        pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                        pred_info['mask'] = clusters.cpu().numpy()
                        gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                        gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)
                        matches[test_scene_name] = {}
                        matches[test_scene_name]['gt'] = gt2pred
                        matches[test_scene_name]['pred'] = pred2gt

            ##### save files
            start3 = time.time()

            if cfg.save_semantic:
                os.makedirs(os.path.join(result_dir, 'semantic_pred'), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(os.path.join(result_dir, 'semantic_pred', test_scene_name + '.npy'), semantic_np)

            if cfg.save_pt_offsets:
                os.makedirs(os.path.join(result_dir, 'pt_offsets'), exist_ok=True)
                pt_offsets = pt_offsets[-1].cpu().numpy()
                np.save(os.path.join(result_dir, 'pt_offsets', test_scene_name + '.npy'), pt_offsets)

            if cfg.save_pt_shifted_coords:
                os.makedirs(os.path.join(result_dir, 'pt_shifted_coords'), exist_ok=True)
                pt_shifted_coords = preds['pt_shifted_coords']
                pt_shifted_coords = pt_shifted_coords.cpu().numpy()
                np.save(os.path.join(result_dir, 'pt_shifted_coords', test_scene_name + '.npy'), pt_shifted_coords)

            if cfg.save_grid_points:
                os.makedirs(os.path.join(result_dir, 'grid_center_preds'), exist_ok=True)
                os.makedirs(os.path.join(result_dir, 'pt_offsets'), exist_ok=True)
                os.makedirs(os.path.join(result_dir, 'semantic_pred'), exist_ok=True)
                grid_center_preds = grid_center_preds.cpu().numpy()
                pt_offsets = pt_offsets[-1].cpu().numpy()
                semantic_pred = semantic_pred.cpu().numpy()
                np.save(os.path.join(result_dir, 'grid_center_preds', test_scene_name + '.npy'), grid_center_preds)
                np.save(os.path.join(result_dir, 'pt_offsets', test_scene_name + '.npy'), pt_offsets)
                np.save(os.path.join(result_dir, 'semantic_pred', test_scene_name + '.npy'), semantic_pred[-1])

            if(epoch > cfg.prepare_epochs and cfg.save_instance):
                f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    semantic_label = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                    score = cluster_scores[proposal_id]
                    f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format(test_scene_name, proposal_id, semantic_label_idx[semantic_label], score))
                    if proposal_id < nclusters - 1:
                        f.write('\n')
                    np.savetxt(os.path.join(result_dir, 'predicted_masks', test_scene_name + '_%03d.txt' % (proposal_id)), clusters_i, fmt='%d')
                f.close()

            end3 = time.time() - start3
            end = time.time() - start
            start = time.time()

            ##### print
            logger.info("instance iter: {}/{} point_num: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(
                batch['id'][0] + 1, len(dataset.test_data_files), N, end, end1, end3)
            )

        ##### evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches)
            avgs = eval.compute_averages(ap_scores)
            eval.print_results(avgs)

        logger.info(
            'point-wise prediction evaluation results: \n'
            'semantic prediction accuracy: {:.6f}'.format(
                point_evaluations['pt_semantic_eval']['True'].cpu().numpy() / (
                        point_evaluations['pt_semantic_eval']['True'] + point_evaluations['pt_semantic_eval']['False']
                ).cpu().numpy(),
            )
        )

        if preds['semantic'][-1].shape[0] == N:
            logger.info(
                'offset prediction distance: {:6f}'.format(
                    np.mean(point_evaluations['pt_offset_eval'])
                )
            )


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


if __name__ == '__main__':
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    if model_name == 'pointgroup':
        from model.pointgroup import PointGroup as Network
        from model.model_functions import model_fn_decorator
    else:
        print("Error: no model version " + model_name)
        exit(0)
    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    model_fn = model_fn_decorator(test=True)

    ##### load model
    utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, dist=False, f=cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    test(model, model_fn, data_name, cfg.test_epoch)
