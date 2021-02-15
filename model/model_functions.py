import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_scatter import scatter_mean
import sys

sys.path.append('../')

from lib.pointgroup_ops.functions import pointgroup_ops
from model.components import WeightedFocalLoss, CenterLoss, TripletLoss


def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    stuff_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    semantic_centre_criterion = CenterLoss(num_classes=cfg.classes, feat_dim=cfg.m, use_gpu=True)
    instance_triplet_criterion = TripletLoss(margin=cfg.triplet_margin)

    # if cfg.offset_norm_criterion == 'l1':
    #     offset_norm_criterion = nn.SmoothL1Loss().cuda()
    if cfg.offset_norm_criterion == 'l2':
        offset_norm_criterion = nn.MSELoss().cuda()
    elif cfg.offset_norm_criterion == 'triplet':
        offset_norm_criterion = nn.SmoothL1Loss().cuda()
        offset_norm_triplet_criterion = nn.TripletMarginLoss(margin=cfg.triplet_margin, p=cfg.triplet_p).cuda()

    centre_criterion = WeightedFocalLoss(alpha=cfg.focal_loss_alpha, gamma=cfg.focal_loss_gamma).cuda()
    centre_semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    centre_offset_criterion = nn.L1Loss().cuda()

    centre_query_criterion = nn.BCEWithLogitsLoss().cuda()

    score_criterion = nn.BCELoss(reduction='none').cuda()
    confidence_criterion = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch):
        coords = batch['point_locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()  # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch['point_coords'].cuda()  # (N, 3), float32, cuda
        feats = batch['point_feats'].cuda()  # (N, C), float32, cuda
        point_positional_encoding = batch['point_positional_encoding'].cuda()

        batch_offsets = batch['batch_offsets'].cuda()  # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        rgb = feats

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float[:, :3]), -1)

        if not cfg.use_ori_coords:
            ori_coords = coords_float[:, 3:]
        else:
            ori_coords = coords_float[:, :3]

        coords_float = coords_float[:, :3]

        # voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda
        #
        # input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ### only be used during debugging
        instance_info = batch['point_instance_infos'].squeeze(dim=0).cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        point_semantic_labels = batch['point_semantic_labels'].squeeze(dim=0).cuda()  # (N), long, cuda

        point_offset_labels = instance_info[:, 0:3] - coords_float

        point_semantic_scores = []
        point_semantic_labels[point_semantic_labels == -100] = 0
        point_semantic_scores.append(torch.nn.functional.one_hot(point_semantic_labels, num_classes=20))

        input_ = {
            'pt_feats': feats,
            'v2p_map': v2p_map,
            'mode': cfg.mode,
            'voxel_coords': voxel_coords.int(),
            'spatial_shape': spatial_shape,
            'batch_size': cfg.batch_size,
            'point_offset_preds': point_offset_labels,
            'point_semantic_scores': point_semantic_scores,
            'point_locs': coords,
            'test': True,
        }

        model.eval()

        ret = model(
            input_, p2v_map, coords_float, rgb, ori_coords, point_positional_encoding,
            coords[:, 0].int(), batch_offsets, epoch
        )

        point_offset_preds = ret['point_offset_preds']
        point_semantic_scores = ret['point_semantic_scores']

        if 'proposal_scores' in ret.keys():
            scores, proposals_idx, proposals_offset = ret['proposal_scores']

        if 'centre_preds' in ret.keys():
            centre_preds = ret['centre_preds']
            centre_preds = centre_preds.squeeze(dim=-1)

            centre_semantic_preds = ret['centre_semantic_preds']
            centre_semantic_preds = centre_semantic_preds.squeeze(dim=0)
            centre_semantic_preds = centre_semantic_preds.max(dim=1)[1]

            centre_offset_preds = ret['centre_offset_preds'] # (B, 32**3, 3)

        ### only be used during debugging
        # instance_info = batch['point_instance_infos'].squeeze(dim=0).cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        # labels = batch['point_semantic_labels'].squeeze(dim=0).cuda()  # (N), long, cuda
        # grid_centre_gt = batch['grid_centre_gt'].squeeze(dim=0).cuda()
        # grid_centre_offset = batch['grid_centre_offset'].squeeze(dim=0).cuda()
        # grid_instance_label = batch['grid_instance_label'].squeeze(dim=0).cuda()

        # point_offset_preds = instance_info[:, 0:3] - coords_float
        #
        # point_semantic_scores = []
        # labels[labels == -100] = 0
        # point_semantic_scores.append(torch.nn.functional.one_hot(labels))

        # fake_grid_centre = torch.zeros_like(grid_centre_preds)
        # fake_grid_centre[0, grid_centre_gt.long()] = 1
        # grid_centre_preds = fake_grid_centre
        #
        # grid_centre_offset_preds[grid_centre_gt.long(), :] = grid_centre_offset
        #
        # grid_centre_semantic_preds = grid_instance_label
        # grid_instance_label[grid_instance_label == -100] = 20

        ##### preds
        with torch.no_grad():
            preds = {}
            preds['pt_offsets'] = point_offset_preds
            preds['semantic'] = point_semantic_scores
            if 'centre_preds' in ret.keys():
                preds['centre_preds'] = centre_preds
                preds['centre_semantic_preds'] = centre_semantic_preds
                preds['centre_offset_preds'] = centre_offset_preds
            preds['pt_coords'] = coords_float
            if (epoch == cfg.test_epoch) and ('proposal_scores' in ret.keys()):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)
            if 'stuff_preds' in ret.keys():
                preds['stuff_preds'] = ret['stuff_preds']
            if 'point_semantic_pred_full' in ret.keys():
                preds['point_semantic_pred_full'] = ret['point_semantic_pred_full']

            preds['pt_semantic_labels'] = point_semantic_labels
            preds['pt_offset_labels'] = point_offset_labels

            if point_offset_preds[-1].shape[0] == coords_float.shape[0]:
                preds['pt_shifted_coords'] = point_offset_preds[-1] + coords_float

        return preds

    def model_fn(batch, model, epoch):
        ##### prepare input and forward
        coords = batch['point_locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()  # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch['point_coords'].cuda()  # (N, 6), float32, cuda
        feats = batch['point_feats'].cuda()  # (N, C), float32, cuda
        labels = batch['point_semantic_labels'].cuda()  # (N), long, cuda
        instance_labels = batch['point_instance_labels'].cuda()  # (N), long, cuda, 0~total_nInst, -100
        point_positional_encoding = batch['point_positional_encoding'].cuda()

        instance_info = batch['point_instance_infos'].cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda
        instance_centres = batch['instance_centres'].cuda()

        batch_offsets = batch['batch_offsets'].cuda()  # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        rgb = feats

        if cfg.use_coords == True:
            feats = torch.cat((feats, coords_float[:, :3]), -1)
        elif cfg.use_coords == 'Z':
            feats = torch.cat((feats, coords_float[:, 2].unsqueeze(dim=-1)), -1)

        if not cfg.use_ori_coords:
            ori_coords = coords_float[:, 3:]
        else:
            ori_coords = coords_float[:, :3]

        coords_float = coords_float[:, :3]

        input_ = {
            'pt_feats': feats,
            'v2p_map': v2p_map,
            'mode': cfg.mode,
            'voxel_coords': voxel_coords.int(),
            'spatial_shape': spatial_shape,
            'batch_size': cfg.batch_size,
            'point_locs': coords,
            'test': False,
        }

        if 'Centre' in cfg.model_mode.split('_'):
            instance_heatmap = batch['grid_centre_heatmap'].cuda()
            grid_centre_gt = batch['grid_centre_indicator'].cuda()
            centre_offset_labels = batch['grid_centre_offset_labels'].cuda()
            centre_semantic_labels = batch['grid_centre_semantic_labels'].cuda()

            centre_queries_coords = batch['centre_queries_coords'].cuda()
            centre_queries_probs = batch['centre_queries_probs'].cuda()
            centre_queries_semantic_labels = batch['centre_queries_semantic_labels'].cuda()
            centre_queries_offsets = batch['centre_queries_offsets'].cuda()
            centre_queries_batch_offsets = batch['centre_queries_batch_offsets'].cuda()

            input_['centre_queries_coords'] = centre_queries_coords
            input_['centre_queries_batch_offsets'] = centre_queries_batch_offsets

        if 'stuff' in cfg.model_mode.split('_'):
            nonstuff_feats = feats[labels > 1]
            nonstuff_spatial_shape = batch['nonstuff_spatial_shape']
            nonstuff_voxel_locs = batch['nonstuff_voxel_locs'].cuda()
            nonstuff_p2v_map = batch['nonstuff_p2v_map'].cuda()
            nonstuff_v2p_map = batch['nonstuff_v2p_map'].cuda()

            input_['nonstuff_feats'] = nonstuff_feats
            input_['nonstuff_spatial_shape'] = nonstuff_spatial_shape
            input_['nonstuff_voxel_locs'] = nonstuff_voxel_locs.int()
            input_['nonstuff_p2v_map'] = nonstuff_p2v_map
            input_['nonstuff_v2p_map'] = nonstuff_v2p_map

        ret = model(
            input_, p2v_map, coords_float, rgb, ori_coords, point_positional_encoding,
            coords[:, 0].int(), batch_offsets, epoch
        )

        point_offset_preds = ret['point_offset_preds']
        point_semantic_scores = ret['point_semantic_scores']

        if 'centre_preds' in ret.keys():
            centre_preds = ret['centre_preds'] # (B, 32**3)
            centre_preds = centre_preds.squeeze(dim=-1)

            centre_semantic_preds = ret['centre_semantic_preds']

            centre_offset_preds = []
            centre_offset_pred = ret['centre_offset_preds'] # (B, 32**3, 3)
            for sample_indx in range(len(batch_offsets) - 1):
                centre_offset_preds.append(
                    centre_offset_pred[sample_indx, grid_centre_gt[grid_centre_gt[:, 0] == sample_indx, :][:, 1]]
                )
            centre_offset_preds = torch.cat(centre_offset_preds, 0) # (nInst, 3)

        if (epoch > cfg.prepare_epochs) and (cfg.model_mode == 'Jiang_original_PointGroup'):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu

        loss_inp = {}

        if point_offset_preds[0].shape[0] == coords_float.shape[0]:
            loss_inp['pt_offsets'] = (
                point_offset_preds, coords_float, instance_info, instance_centres, instance_labels
            )
            loss_inp['semantic_scores'] = (point_semantic_scores, labels)

        elif point_offset_preds[0].shape[0] == nonstuff_feats.shape[0]:
            loss_inp['pt_offsets'] = (
                point_offset_preds, coords_float[labels > 1], instance_info[labels > 1],
                instance_centres, instance_labels[labels > 1]
            )
            loss_inp['semantic_scores'] = (point_semantic_scores, labels[labels > 1])

        if 'centre_preds' in ret.keys():
            loss_inp['centre_preds'] = (centre_preds, instance_heatmap)
            loss_inp['centre_semantic_preds'] = (centre_semantic_preds, centre_semantic_labels)
            loss_inp['centre_offset_preds'] = (centre_offset_preds, centre_offset_labels[:, 1:])

        if (epoch > cfg.prepare_epochs) and ('proposal_scores' in ret.keys()):
            loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)

        if 'proposal_confidences' in ret.keys():
            proposals_confidence_preds, proposals_idx_shifts, proposals_offset_shifts = ret['proposal_confidences']
            loss_inp['proposal_confidences'] = (
                proposals_confidence_preds, proposals_idx_shifts, proposals_offset_shifts, instance_pointnum
            )

        if 'stuff_preds' in ret.keys():
            stuff_preds = ret['stuff_preds']
            loss_inp['stuff_preds'] = (stuff_preds, labels)

        if 'output_feats' in ret.keys():
            stuff_output_feats = ret['output_feats']
            loss_inp['output_feats'] = (stuff_output_feats, labels)

        if 'queries_preds' in ret.keys():
            queries_preds = ret['queries_preds']
            queries_semantic_preds = ret['queries_semantic_preds']
            queries_offset_preds = ret['queries_offset_preds']

            loss_inp['queries_preds'] = (queries_preds.squeeze(dim=-1), centre_queries_probs)
            loss_inp['queries_semantic_preds'] = (queries_semantic_preds, centre_queries_semantic_labels)
            loss_inp['queries_offset_preds'] = (queries_offset_preds, centre_queries_offsets)

        if 'points_semantic_center_loss_feature' in ret.keys():
            points_semantic_center_loss_feature = ret['points_semantic_center_loss_feature']

            loss_inp['points_semantic_center_loss_feature'] = (points_semantic_center_loss_feature, instance_labels, labels)

        if cfg.instance_triplet_loss and 'point_offset_feats' in ret.keys():
            point_offset_feats = ret['point_offset_feats']

            loss_inp['point_offset_feats'] = (point_offset_feats, instance_labels, batch_offsets)

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['pt_offsets'] = point_offset_preds
            preds['semantic_scores'] = point_semantic_scores
            if 'centre_preds' in ret.keys():
                preds['centre_preds'] = centre_preds
                preds['centre_semantic_preds'] = centre_semantic_preds
                preds['centre_offset_preds'] = centre_offset_preds
            if (epoch > cfg.prepare_epochs) and ('proposal_scores' in ret.keys()):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, preds, visual_dict, meter_dict

    def loss_fn(loss_inp, epoch):

        loss_out = {}
        infos = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = torch.zeros(1).cuda()
        for semantic_score in semantic_scores:
            if (cfg.model_mode == 'Yu_stuff_sep_PointGroup') or (cfg.model_mode == 'Yu_stuff_remove_PointGroup'):
                nonstuff_indx = (semantic_labels > 1)
                nonstuff_semantic_labels = semantic_labels[nonstuff_indx] - 2
                semantic_loss += semantic_criterion(semantic_score[nonstuff_indx], nonstuff_semantic_labels)
            else:
                semantic_loss += semantic_criterion(semantic_score, semantic_labels)

        loss_out['semantic_loss'] = (semantic_loss, semantic_scores[0].shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_centre, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        offset_norm_loss = torch.zeros(1).cuda()
        offset_dir_loss = torch.zeros(1).cuda()
        for pt_offset in pt_offsets:
            gt_offsets = instance_info[:, 0:3] - coords  # (N, 3)
            pt_valid_index = instance_labels != cfg.ignore_label
            if cfg.offset_norm_criterion == 'l1':
                pt_diff = pt_offset - gt_offsets  # (N, 3)
                pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
                valid = (instance_labels != cfg.ignore_label).float()
                offset_norm_loss += torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)
            if cfg.offset_norm_criterion == 'l2':
                offset_norm_loss += offset_norm_criterion(pt_offset[pt_valid_index], gt_offsets[pt_valid_index])
            elif cfg.offset_norm_criterion == 'triplet':
                ### offset l1 distance loss: learn to move towards the true instance centre
                offset_norm_loss += offset_norm_criterion(pt_offset[pt_valid_index], gt_offsets[pt_valid_index])

                # positive_offset = instance_info[:, 0:3].unsqueeze(dim=1).repeat(1, instance_centre.shape[0], 1)
                # negative_offset = instance_centre.unsqueeze(dim=0).repeat(coords.shape[0], 1, 1)
                # positive_offset_index = (negative_offset == positive_offset).to(torch.bool) == torch.ones(3, dtype=torch.bool).cuda()
                # negative_offset[positive_offset_index] = 100
                # negative_offset = negative_offset - positive_offset
                # negative_offset_index = torch.norm(
                #     gt_offsets.unsqueeze(dim=1).repeat(1, instance_centre.shape[0], 1) - negative_offset, dim=2
                # ).min(dim=1)[1][:, 0]

                ### offset triplet loss: learn to leave the second closest point
                ## semi-hard negative sampling
                shifted_coords = coords + pt_offset
                positive_offsets = shifted_coords - instance_info[:, 0:3]
                positive_distance = torch.norm(positive_offsets, dim=1)
                negative_offsets = shifted_coords.unsqueeze(dim=1).repeat(1, instance_centre.shape[0], 1) - \
                                   instance_centre.unsqueeze(dim=0).repeat(shifted_coords.shape[0], 1, 1)
                negative_distance = torch.norm(negative_offsets, dim=2)
                negative_offset_index = (
                        negative_distance - positive_distance.unsqueeze(dim=1).repeat(1, instance_centre.shape[0])
                ).min(dim=1)[1]
                semi_negative_sample_index = torch.ones(shifted_coords.shape[0], dtype=torch.bool).cuda()
                # ignore points whose distance from the second closest point is larger than the triplet margin
                semi_negative_sample_index[
                    (negative_distance.topk(2, dim=1, largest=False)[0][:, -1] - positive_distance) > cfg.triplet_margin] = 0
                semi_negative_sample_index[
                    (negative_distance.topk(2, dim=1, largest=False)[0][:, -1] - positive_distance) < 0] = 0
                # 1 st false
                semi_negative_sample_index[
                    (negative_distance.topk(1, dim=1, largest=False)[0][:, -1] - positive_distance) != 0] = 1
                # ignore points with -100 instance label
                semi_negative_sample_index[instance_labels == cfg.ignore_label] = 0

                ### calculate triplet loss based predicted offset vector and gt offset vector
                # negative_offset = instance_centre[negative_offset_index] - coords
                # offset_norm_triplet_loss = offset_norm_triplet_criterion(
                #     pt_offsets[semi_negative_sample_index], gt_offsets[semi_negative_sample_index],
                #     negative_offset[semi_negative_sample_index]
                # )

                ### calculate triplet loss based shifted coordinates and centre coordinates
                negative_coords = instance_centre[negative_offset_index]
                gt_coords = instance_info[:, 0:3]
                offset_norm_triplet_loss = offset_norm_triplet_criterion(
                    shifted_coords[semi_negative_sample_index], gt_coords[semi_negative_sample_index],
                    negative_coords[semi_negative_sample_index]
                )

                if torch.isnan(offset_norm_triplet_loss):
                    offset_norm_triplet_loss = torch.tensor(0, dtype=torch.float).cuda()
                loss_out['offset_norm_triplet_loss'] = (offset_norm_triplet_loss, semi_negative_sample_index.to(torch.float32).sum())

            # if cfg.constrastive_loss:
            #     shifted_coords = coords + pt_offsets


            # pt_diff = pt_offsets - gt_offsets  # (N, 3)
            # pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
            valid = (instance_labels != cfg.ignore_label).float()
            # offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

            gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
            gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
            pt_offsets_norm = torch.norm(pt_offset, p=2, dim=1)
            pt_offsets_ = pt_offset / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
            direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
            offset_dir_loss += torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        if 'centre_preds' in loss_inp.keys():
            '''centre loss'''
            centre_preds, instance_heatmap = loss_inp['centre_preds']

            centre_loss = centre_criterion(centre_preds, instance_heatmap)
            loss_out['centre_loss'] = (centre_loss, instance_heatmap.shape[-1])

            '''centre semantic loss'''
            centre_semantic_preds, grid_instance_label = loss_inp['centre_semantic_preds']
            # grid_valid_index = instance_heatmap > 0
            # centre_semantic_loss = []
            # for sample_index in range(instance_heatmap.shape[0]):
            #     if sum(grid_valid_index[sample_index, :]) != 0:
            #         centre_semantic_loss.append(
            #             centre_semantic_criterion(
            #                 centre_semantic_preds[sample_index, grid_valid_index[sample_index, :], :],
            #                 grid_instance_label[sample_index, grid_valid_index[sample_index, :]].to(torch.long)
            #             )
            #         )
            # if len(centre_semantic_loss) != 0:
            #     centre_semantic_loss = torch.mean(torch.stack(centre_semantic_loss))
            # else:
            #     centre_semantic_loss = 0
            centre_semantic_loss = centre_semantic_criterion(
                centre_semantic_preds.reshape(-1, centre_semantic_preds.shape[-1]), grid_instance_label.reshape(-1))

            loss_out['centre_semantic_loss'] = (centre_semantic_loss, grid_instance_label.shape[0])

            '''centre offset loss'''
            centre_offset_preds, grid_centre_offsets = loss_inp['centre_offset_preds']

            centre_offset_loss = centre_offset_criterion(centre_offset_preds, grid_centre_offsets)
            loss_out['centre_offset_loss'] = (centre_offset_loss, grid_centre_offsets.shape[0])

        if (epoch > cfg.prepare_epochs) and ('proposal_scores' in ret.keys()):
            '''score loss'''
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp['proposal_scores']
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels, instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

            score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            loss_out['score_loss'] = (score_loss, gt_ious.shape[0])

        if 'proposal_confidences' in loss_inp.keys():
            proposals_confidence_preds, proposals_idx_shifts, proposals_offset_shifts, instance_pointnum = loss_inp['proposal_confidences']

            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            confidence_loss = torch.zeros(1).cuda()

            for proposal_index in range(len(proposals_confidence_preds)):
                ious = pointgroup_ops.get_iou(
                    proposals_idx_shifts[proposal_index][:, 1].cuda(), proposals_offset_shifts[proposal_index].cuda(),
                    instance_labels, instance_pointnum
                ) # (nProposal, nInstance), float
                gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
                gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

                conf_loss = confidence_criterion(torch.sigmoid(proposals_confidence_preds[proposal_index].view(-1)),
                                                 gt_scores)
                confidence_loss += conf_loss.mean()

            loss_out['score_loss'] = (confidence_loss, gt_ious.shape[0])

        if 'stuff_preds' in loss_inp.keys():
            stuff_prediction, semantic_labels = loss_inp['stuff_preds']
            stuff_labels = torch.zeros(stuff_prediction.shape[0]).long().cuda()
            stuff_labels[semantic_labels > 1] = 1
            stuff_loss = stuff_criterion(stuff_prediction, stuff_labels)
            stuff_loss = (cfg.focal_loss_alpha * (1 - torch.exp(-stuff_loss)) ** cfg.focal_loss_gamma * stuff_loss).mean()  # mean over the batch
            loss_out['stuff_loss'] = (stuff_loss, stuff_prediction.shape[0])

        if 'output_feats' in loss_inp.keys():
            output_feats, semantic_labels = loss_inp['output_feats']
            stuff_indx = (semantic_labels < 2)
            stuff_output_feats = output_feats[stuff_indx]
            stuff_feats_norm_loss = torch.norm(torch.mean(stuff_output_feats, dim=0), p=2, dim=0)

        ### calculate loss related with centre queries
        if 'queries_preds' in loss_inp.keys():
            queries_preds, centre_queries_probs = loss_inp['queries_preds']
            centre_queries_loss = centre_query_criterion(queries_preds, centre_queries_probs)
            loss_out['centre_loss'] = (centre_queries_loss, centre_queries_probs.shape[-1])

            queries_semantic_preds, centre_queries_semantic_labels = loss_inp['queries_semantic_preds']
            centre_queries_semantic_loss = centre_semantic_criterion(
                queries_semantic_preds.reshape(-1, queries_semantic_preds.shape[-1]),
                centre_queries_semantic_labels.reshape(-1)
            )
            loss_out['centre_semantic_loss'] = (centre_queries_semantic_loss, centre_queries_semantic_labels.shape[0])

            queries_offset_preds, centre_queries_offsets = loss_inp['queries_offset_preds']
            centre_queries_offset_loss = centre_offset_criterion(queries_offset_preds, centre_queries_offsets)
            loss_out['centre_offset_loss'] = (centre_queries_offset_loss, centre_queries_offsets.shape[0])

        if 'points_semantic_center_loss_feature' in loss_inp.keys():
            points_semantic_center_loss_features, instance_labels, labels = loss_inp['points_semantic_center_loss_feature']

            semantic_centre_loss = torch.zeros(1).cuda()
            for points_semantic_center_loss_feature in points_semantic_center_loss_features:
                semantic_centre_loss += semantic_centre_criterion(points_semantic_center_loss_feature, labels)

            loss_out['centre_offset_loss'] = (semantic_centre_loss, labels.shape[0])

        if cfg.instance_triplet_loss and 'point_offset_feats' in loss_inp.keys():
            point_offset_feats, instance_labels, batch_offsets = loss_inp['point_offset_feats']

            instance_triplet_loss = torch.zeros(1).cuda()
            for batch_id in range(1, len(batch_offsets)):
                point_offset_feat = point_offset_feats[batch_offsets[batch_id - 1]:batch_offsets[batch_id]]
                instance_label = instance_labels[batch_offsets[batch_id - 1]:batch_offsets[batch_id]]
                valid_instance_index = (instance_label != cfg.ignore_label)
                point_offset_feat = point_offset_feat[valid_instance_index]
                instance_label = instance_label[valid_instance_index]

                triplet_index = []
                for inst_id in instance_label.unique():
                    triplet_index.append(
                        torch.cat(random.sample(list((instance_label == inst_id).nonzero()),
                                                min((instance_label == inst_id).sum().item(),
                                                    cfg.instance_triplet_loss_sample_point_num))))

                triplet_index = torch.cat(triplet_index)

                instance_triplet_loss += instance_triplet_criterion(
                    point_offset_feat[triplet_index], instance_label[triplet_index])[0]

            instance_triplet_loss = instance_triplet_loss / (len(batch_offsets) - 1)

            loss_out['centre_offset_loss'] = (instance_triplet_loss, instance_labels.shape[0])


        '''total loss'''
        loss = cfg.loss_weight[3] * semantic_loss + cfg.loss_weight[4] * offset_norm_loss + \
               cfg.loss_weight[5] * offset_dir_loss
        if cfg.offset_norm_criterion == 'triplet':
            loss += cfg.loss_weight[6] * offset_norm_triplet_loss
        if 'centre_preds' in loss_inp.keys():
            loss += cfg.loss_weight[0] * centre_loss + cfg.loss_weight[1] * centre_semantic_loss + \
                    cfg.loss_weight[2] * centre_offset_loss
        if (epoch > cfg.prepare_epochs) and ('proposal_scores' in ret.keys()):
            loss += (cfg.loss_weight[7] * score_loss)
        if 'proposal_confidences' in loss_inp.keys():
            loss += confidence_loss
        if 'stuff_preds' in loss_inp.keys():
            loss += stuff_loss
        if 'output_feats' in loss_inp.keys():
            loss += cfg.loss_weight[8] * stuff_feats_norm_loss

        if 'queries_preds' in loss_inp.keys():
            loss = loss + centre_queries_loss + centre_queries_semantic_loss + centre_queries_offset_loss

        if cfg.instance_triplet_loss and 'point_offset_feats' in loss_inp.keys():
            loss += instance_triplet_loss

        return loss, loss_out, infos

    def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
        '''
        :param scores: (N), float, 0~1
        :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
        '''
        fg_mask = scores > fg_thresh
        bg_mask = scores < bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        segmented_scores = (fg_mask > 0).float()
        k = 1 / (fg_thresh - bg_thresh)
        b = bg_thresh / (bg_thresh - fg_thresh)
        segmented_scores[interval_mask] = scores[interval_mask] * k + b

        return segmented_scores

    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn
