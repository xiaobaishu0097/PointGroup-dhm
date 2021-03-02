import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_scatter import scatter_mean
import sys

sys.path.append('../')

from lib.pointgroup_ops.functions import pointgroup_ops
from model.components import WeightedFocalLoss, CenterLoss, TripletLoss
from model.common import generate_adaptive_heatmap


def model_fn_decorator(cfg, test=False):
    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    stuff_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    semantic_center_criterion = CenterLoss(num_classes=cfg.classes, feat_dim=cfg.m, use_gpu=True)
    instance_triplet_criterion = TripletLoss(margin=cfg.instance_triplet_loss['margin'])

    if cfg.offset_norm_criterion == 'l2':
        offset_norm_criterion = nn.MSELoss().cuda()

    center_criterion = WeightedFocalLoss(alpha=cfg.focal_loss['alpha'], gamma=cfg.focal_loss['gamma']).cuda()
    center_semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    center_offset_criterion = nn.L1Loss().cuda()

    center_query_criterion = nn.BCEWithLogitsLoss().cuda()

    score_criterion = nn.BCELoss(reduction='none').cuda()
    confidence_criterion = nn.BCELoss(reduction='none').cuda()

    point_reconstruction_criterion = nn.SmoothL1Loss().cuda()
    point_instance_id_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    def test_model_fn(batch, model, epoch):
        coords = batch['point_locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()  # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch['point_coords'].cuda()  # (N, 3), float32, cuda
        feats = batch['point_feats'].cuda()  # (N, C), float32, cuda

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
            input_, p2v_map, coords_float, rgb, ori_coords,
            coords[:, 0].int(), batch_offsets, epoch
        )

        point_offset_preds = ret['point_offset_preds']
        point_semantic_scores = ret['point_semantic_scores']

        if 'proposal_scores' in ret.keys():
            scores, proposals_idx, proposals_offset = ret['proposal_scores']

        if 'center_preds' in ret.keys():
            center_preds, sampled_index = ret['center_preds']
            center_preds = center_preds.squeeze(dim=-1)

            center_semantic_preds, _ = ret['center_semantic_preds']
            center_semantic_preds = center_semantic_preds.squeeze(dim=0)
            center_semantic_preds = center_semantic_preds.max(dim=1)[1]

            center_offset_preds, _ = ret['center_offset_preds'] # (B, 32**3, 3)

        ### only be used during debugging
        # instance_info = batch['point_instance_infos'].squeeze(dim=0).cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        # labels = batch['point_semantic_labels'].squeeze(dim=0).cuda()  # (N), long, cuda
        # grid_center_gt = batch['grid_center_gt'].squeeze(dim=0).cuda()
        # grid_center_offset = batch['grid_center_offset'].squeeze(dim=0).cuda()
        # grid_instance_label = batch['grid_instance_label'].squeeze(dim=0).cuda()

        # point_offset_preds = instance_info[:, 0:3] - coords_float
        #
        # point_semantic_scores = []
        # labels[labels == -100] = 0
        # point_semantic_scores.append(torch.nn.functional.one_hot(labels))

        # fake_grid_center = torch.zeros_like(grid_center_preds)
        # fake_grid_center[0, grid_center_gt.long()] = 1
        # grid_center_preds = fake_grid_center
        #
        # grid_center_offset_preds[grid_center_gt.long(), :] = grid_center_offset
        #
        # grid_center_semantic_preds = grid_instance_label
        # grid_instance_label[grid_instance_label == -100] = 20

        ##### preds
        with torch.no_grad():
            preds = {}
            preds['pt_offsets'] = point_offset_preds
            preds['semantic'] = point_semantic_scores
            if 'center_preds' in ret.keys():
                preds['center_preds'] = center_preds
                preds['center_semantic_preds'] = center_semantic_preds
                preds['center_offset_preds'] = center_offset_preds
            preds['pt_coords'] = coords_float
            if (epoch == cfg.test_epoch) and ('proposal_scores' in ret.keys()):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)
            if 'stuff_preds' in ret.keys():
                preds['stuff_preds'] = ret['stuff_preds']
            if 'point_semantic_pred_full' in ret.keys():
                preds['point_semantic_pred_full'] = ret['point_semantic_pred_full']
            if ('occupancy' in cfg.model_mode.split('_')) and ('voxel_occupancy_preds' in ret.keys()):
                preds['point_occupancy_preds'] = ret['voxel_occupancy_preds'][0][p2v_map.long()]

                voxel_instance_labels = batch['voxel_instance_labels'].cuda()
                voxel_occupancy_labels = batch['voxel_occupancy_labels'].cuda()

                preds['point_instance_labels'] = voxel_instance_labels[p2v_map.long()]
                preds['point_occupancy_labels'] = voxel_occupancy_labels[p2v_map.long()]
            if cfg.model_mode == 'Center_pointnet++_clustering':
                preds['sampled_index'] = sampled_index

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

        instance_info = batch['point_instance_infos'].cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda
        instance_centers = batch['instance_centers'].cuda()
        instance_sizes = batch['instance_sizes'].cuda()

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
            'semantic_labels': labels,
        }

        if 'Center' in cfg.model_mode.split('_'):
            instance_heatmap = batch['grid_center_heatmap'].cuda()
            grid_center_gt = batch['grid_center_indicator'].cuda()
            center_offset_labels = batch['grid_center_offset_labels'].cuda()
            center_semantic_labels = batch['grid_center_semantic_labels'].cuda()

            center_queries_coords = batch['center_queries_coords'].cuda()
            center_queries_probs = batch['center_queries_probs'].cuda()
            center_queries_semantic_labels = batch['center_queries_semantic_labels'].cuda()
            center_queries_offsets = batch['center_queries_offsets'].cuda()
            center_queries_batch_offsets = batch['center_queries_batch_offsets'].cuda()

            input_['center_queries_coords'] = center_queries_coords
            input_['center_queries_batch_offsets'] = center_queries_batch_offsets

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

        if ('occupancy' in cfg.model_mode.split('_')):
            voxel_instance_labels = batch['voxel_instance_labels'].cuda()
            voxel_occupancy_labels = batch['voxel_occupancy_labels'].cuda()

        ret = model(
            input_, p2v_map, coords_float, rgb, ori_coords,
            coords[:, 0].int(), batch_offsets, epoch
        )

        point_offset_preds = ret['point_offset_preds']
        point_semantic_scores = ret['point_semantic_scores']

        loss_inp = {}

        if point_offset_preds[0].shape[0] == coords_float.shape[0]:
            loss_inp['pt_offsets'] = (
                point_offset_preds, coords_float, instance_info, instance_centers, instance_labels
            )
            loss_inp['semantic_scores'] = (point_semantic_scores, labels)

        elif point_offset_preds[0].shape[0] == nonstuff_feats.shape[0]:
            loss_inp['pt_offsets'] = (
                point_offset_preds, coords_float[labels > 1], instance_info[labels > 1],
                instance_centers, instance_labels[labels > 1]
            )
            loss_inp['semantic_scores'] = (point_semantic_scores, labels[labels > 1])

        if 'center_preds' in ret.keys():
            if not isinstance(ret['center_preds'], tuple):
                center_preds = ret['center_preds'] # (B, 32**3)
                center_preds = center_preds.squeeze(dim=-1)

                center_semantic_preds = ret['center_semantic_preds']

                center_offset_preds = []
                center_offset_pred = ret['center_offset_preds'] # (B, 32**3, 3)
                for sample_indx in range(len(batch_offsets) - 1):
                    center_offset_preds.append(
                        center_offset_pred[sample_indx, grid_center_gt[grid_center_gt[:, 0] == sample_indx, :][:, 1]]
                    )
                center_offset_preds = torch.cat(center_offset_preds, 0) # (nInst, 3)

                loss_inp['center_preds'] = (center_preds, instance_heatmap)
                loss_inp['center_semantic_preds'] = (center_semantic_preds, center_semantic_labels)
                loss_inp['center_offset_preds'] = (center_offset_preds, center_offset_labels[:, 1:])
            else:
                center_preds, sampled_indexes = ret['center_preds'] # (B, 8196, 1)
                center_semantic_preds, _ = ret['center_semantic_preds'] # (B, 8196, 20)
                center_offset_preds, _ = ret['center_offset_preds'] # (B, 8196, 3)

                loss_inp['center_preds'] = (
                    center_preds, coords_float, sampled_indexes, instance_centers, instance_sizes, batch_offsets
                )
                loss_inp['center_semantic_preds'] = (center_semantic_preds, sampled_indexes, labels)
                loss_inp['center_offset_preds'] = (
                    center_offset_preds, sampled_indexes, coords_float, instance_info, instance_labels
                )

        if (epoch > cfg.prepare_epochs) and ('proposal_scores' in ret.keys()):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu

            loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)

        if 'proposal_confidences' in ret.keys():
            proposals_confidence_preds, proposals_idx_shifts, proposals_offset_shifts = ret['proposal_confidences']
            loss_inp['proposal_confidences'] = (
                proposals_confidence_preds, proposals_idx_shifts, proposals_offset_shifts, instance_pointnum
            )

        if 'stuff_preds' in ret.keys():
            stuff_preds = ret['stuff_preds']
            loss_inp['stuff_preds'] = (stuff_preds, labels)

        if cfg.stuff_norm_loss['activate'] and 'output_feats' in ret.keys():
            stuff_output_feats = ret['output_feats']
            loss_inp['output_feats'] = (stuff_output_feats, labels)

        if 'queries_preds' in ret.keys():
            queries_preds = ret['queries_preds']
            queries_semantic_preds = ret['queries_semantic_preds']
            queries_offset_preds = ret['queries_offset_preds']

            loss_inp['queries_preds'] = (queries_preds.squeeze(dim=-1), center_queries_probs)
            loss_inp['queries_semantic_preds'] = (queries_semantic_preds, center_queries_semantic_labels)
            loss_inp['queries_offset_preds'] = (queries_offset_preds, center_queries_offsets)

        if 'points_semantic_center_loss_feature' in ret.keys():
            points_semantic_center_loss_feature = ret['points_semantic_center_loss_feature']

            loss_inp['points_semantic_center_loss_feature'] = (points_semantic_center_loss_feature, labels, batch_offsets)

        if cfg.instance_triplet_loss['activate'] and 'point_offset_feats' in ret.keys():
            point_offset_feats = ret['point_offset_feats']

            loss_inp['point_offset_feats'] = (point_offset_feats, instance_labels, batch_offsets)

        ### try three different feature term losses mentioned in OccuSeg
        if cfg.feature_variance_loss['activate'] and ('point_features' in ret.keys()):
            point_features = ret['point_features']
            loss_inp['feature_variance_loss'] = (point_features, instance_labels)

        if cfg.feature_distance_loss['activate'] and ('point_features' in ret.keys()):
            point_features = ret['point_features']
            loss_inp['feature_distance_loss'] = (point_features, instance_labels)

        if cfg.feature_instance_regression_loss['activate'] and ('point_features' in ret.keys()):
            point_features = ret['point_features']
            loss_inp['feature_instance_regression_loss'] = (point_features, instance_labels)

        if cfg.feature_semantic_regression_loss['activate'] and ('point_features' in ret.keys()):
            point_features = ret['point_features']
            loss_inp['feature_semantic_regression_loss'] = (point_features, labels)

        ### occupancy loss to predict the voxel number for each voxel
        if ('occupancy' in cfg.model_mode.split('_')) and ('voxel_occupancy_preds' in ret.keys()):
            voxel_occupancy_preds = ret['voxel_occupancy_preds']

            loss_inp['voxel_occupancy_loss'] = (voxel_occupancy_preds, voxel_instance_labels, voxel_occupancy_labels)

        if ('local_point_semantic_scores' in ret.keys()) and (cfg.local_proposal['scatter_mean_target'] == False):
            local_point_semantic_scores, local_proposals_idx = ret['local_point_semantic_scores']
            local_point_offset_preds, _ = ret['local_point_offset_preds']

            loss_inp['local_point_semantic'] = (local_point_semantic_scores, local_proposals_idx, labels)
            loss_inp['local_point_offset'] = (local_point_offset_preds, local_proposals_idx, coords_float, instance_info, instance_labels)

        if ('point_reconstructed_coords' in ret.keys()) and (cfg.point_xyz_reconstruction_loss['activate']):
            point_reconstructed_coords = ret['point_reconstructed_coords']

            loss_inp['point_reconstructed_coords'] = (point_reconstructed_coords, coords_float)

        if ('point_reconstructed_colors' in ret.keys()) and (cfg.point_rgb_reconstruction_loss['activate']):
            point_reconstructed_colors = ret['point_reconstructed_colors']

            loss_inp['point_reconstructed_colors'] = (point_reconstructed_colors, rgb)

        if ('instance_id_preds' in ret.keys()) and (cfg.instance_classifier['activate']):
            instance_id_preds = ret['instance_id_preds']

            loss_inp['instance_id_preds'] = (instance_id_preds, instance_labels)

        if ('proposals_point_features' in ret.keys()) and (cfg.local_proposal['local_point_feature_discriminative_loss']):
            proposals_point_features, proposals_idx = ret['proposals_point_features']

            loss_inp['local_point_feature_discriminative_loss'] = (proposals_point_features, proposals_idx, instance_labels)

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['pt_offsets'] = point_offset_preds
            preds['semantic_scores'] = point_semantic_scores
            if 'center_preds' in ret.keys():
                preds['center_preds'] = center_preds
                preds['center_semantic_preds'] = center_semantic_preds
                preds['center_offset_preds'] = center_offset_preds
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

        semantic_loss = torch.zeros(1).cuda(semantic_scores[0].device)
        for semantic_score in semantic_scores:
            if (cfg.model_mode == 'Yu_stuff_sep_PointGroup') or (cfg.model_mode == 'Yu_stuff_remove_PointGroup'):
                nonstuff_indx = (semantic_labels > 1)
                nonstuff_semantic_labels = semantic_labels[nonstuff_indx] - 2
                semantic_loss += semantic_criterion(semantic_score[nonstuff_indx], nonstuff_semantic_labels)
            else:
                semantic_loss += semantic_criterion(semantic_score, semantic_labels.to(semantic_score.device))

        loss_out['semantic_loss'] = (semantic_loss, semantic_scores[0].shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_center, instance_labels = loss_inp['pt_offsets']
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

            valid = (instance_labels != cfg.ignore_label).float()
            gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
            gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
            pt_offsets_norm = torch.norm(pt_offset, p=2, dim=1)
            pt_offsets_ = pt_offset / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
            direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
            offset_dir_loss += torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        if 'center_preds' in loss_inp.keys():
            if len(loss_inp['center_preds']) == 2:
                '''center loss'''
                center_preds, instance_heatmap = loss_inp['center_preds']

                center_loss = center_criterion(center_preds, instance_heatmap)
                loss_out['center_loss'] = (center_loss, instance_heatmap.shape[-1])

                '''center semantic loss'''
                center_semantic_preds, grid_instance_label = loss_inp['center_semantic_preds']
                center_semantic_loss = center_semantic_criterion(
                    center_semantic_preds.reshape(-1, center_semantic_preds.shape[-1]), grid_instance_label.reshape(-1))

                loss_out['center_semantic_loss'] = (center_semantic_loss, grid_instance_label.shape[0])

                '''center offset loss'''
                center_offset_preds, grid_center_offsets = loss_inp['center_offset_preds']

                center_offset_loss = center_offset_criterion(center_offset_preds, grid_center_offsets)
                loss_out['center_offset_loss'] = (center_offset_loss, grid_center_offsets.shape[0])

            # using pointnet++ to predict the center probability of sampled points
            elif len(loss_inp['center_preds']) == 6:
                center_heatmaps = []
                center_semantic_labels = []

                center_offset_norm_loss = torch.zeros(1).cuda()
                center_offset_dir_loss = torch.zeros(1).cuda()

                center_preds, point_coords, sampled_indexes, instance_centers, instance_sizes, batch_offsets = loss_inp['center_preds']
                center_semantic_preds, _, point_semantic_labels = loss_inp['center_semantic_preds']
                center_offset_preds, _, point_coords, point_instance_info, instance_labels = loss_inp['center_offset_preds']

                for batch_index in range(1, len(batch_offsets)):
                    '''center loss'''
                    point_coord = point_coords[batch_offsets[batch_index - 1]:batch_offsets[batch_index]]
                    instance_center = instance_centers[instance_centers[:, 0] == (batch_index - 1), 1:]
                    instance_size = instance_sizes[instance_centers[:, 0] == (batch_index - 1), 1:]

                    center_heatmap = generate_adaptive_heatmap(
                        point_coord[sampled_indexes[batch_index - 1], :].double().cpu(), instance_center.cpu(),
                        instance_size.cpu(), min_IoU=cfg.min_IoU
                    )['heatmap']

                    center_heatmaps.append(center_heatmap.unsqueeze(dim=0).cuda())

                    '''center semantic loss'''
                    point_semantic_label = point_semantic_labels[batch_offsets[batch_index - 1]:batch_offsets[batch_index]]
                    center_semantic_labels.append(point_semantic_label[sampled_indexes[batch_index - 1]].unsqueeze(dim=0))

                    '''center offset loss'''
                    center_instance_info = point_instance_info[batch_offsets[batch_index - 1]:batch_offsets[batch_index]]
                    center_instance_info = center_instance_info[sampled_indexes[batch_index - 1]]
                    center_coord = point_coords[batch_offsets[batch_index - 1]:batch_offsets[batch_index]]
                    center_coord = center_coord[sampled_indexes[batch_index - 1]]
                    center_offset_pred = center_offset_preds[batch_index - 1, :]
                    instance_label = instance_labels[batch_offsets[batch_index - 1]:batch_offsets[batch_index]]
                    instance_label = instance_label[sampled_indexes[batch_index - 1]]

                    gt_offsets = center_instance_info[:, 0:3] - center_coord  # (8196, 3)
                    center_diff = center_offset_pred - gt_offsets  # (N, 3)
                    center_dist = torch.sum(torch.abs(center_diff), dim=-1)  # (N)
                    valid = (instance_label != cfg.ignore_label).float()
                    center_offset_norm_loss += torch.sum(center_dist * valid) / (torch.sum(valid) + 1e-6)

                    gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
                    gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
                    center_offsets_norm = torch.norm(center_offset_pred, p=2, dim=1)
                    center_offsets_ = center_offset_pred / (center_offsets_norm.unsqueeze(-1) + 1e-8)
                    direction_diff = - (gt_offsets_ * center_offsets_).sum(-1)  # (N)
                    center_offset_dir_loss += torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)


                center_heatmaps = torch.cat(center_heatmaps, dim=0).to(torch.float32).unsqueeze(dim=-1)
                center_probs_loss = center_criterion(center_preds, center_heatmaps)

                center_semantic_labels = torch.cat(center_semantic_labels, dim=0)
                batch_size = center_semantic_preds.shape[0]
                center_semantic_loss = center_semantic_criterion(
                    center_semantic_preds.view(batch_size * 8196, 20),
                    center_semantic_labels.view(batch_size * 8196)
                )
                center_offset_norm_loss = center_offset_norm_loss / cfg.batch_size
                center_offset_dir_loss = center_offset_dir_loss / cfg.batch_size

                loss_out['center_probs_loss'] = (center_probs_loss, sampled_indexes.shape[0] * 8196)
                loss_out['center_semantic_loss'] = (center_semantic_loss, sampled_indexes.shape[0] * 8196)
                loss_out['center_offset_norm_loss'] = (center_offset_norm_loss, sampled_indexes.shape[0] * 8196)
                loss_out['center_offset_dir_loss'] = (center_offset_dir_loss, sampled_indexes.shape[0] * 8196)

        if (epoch > cfg.prepare_epochs) and ('proposal_scores' in loss_inp.keys()):
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
            stuff_loss = (cfg.focal_loss['alpha'] * (1 - torch.exp(-stuff_loss)) ** cfg.focal_loss['gamma'] * stuff_loss).mean()  # mean over the batch
            loss_out['stuff_loss'] = (stuff_loss, stuff_prediction.shape[0])

        if cfg.stuff_norm_loss['activate'] and 'output_feats' in loss_inp.keys():
            output_feats, semantic_labels = loss_inp['output_feats']
            stuff_indx = (semantic_labels < 2)
            stuff_output_feats = output_feats[stuff_indx]
            stuff_feats_norm_loss = torch.norm(torch.mean(stuff_output_feats, dim=0), p=2, dim=0)

        ### calculate loss related with center queries
        if 'queries_preds' in loss_inp.keys():
            queries_preds, center_queries_probs = loss_inp['queries_preds']
            center_queries_loss = center_query_criterion(queries_preds, center_queries_probs)
            loss_out['center_loss'] = (center_queries_loss, center_queries_probs.shape[-1])

            queries_semantic_preds, center_queries_semantic_labels = loss_inp['queries_semantic_preds']
            center_queries_semantic_loss = center_semantic_criterion(
                queries_semantic_preds.reshape(-1, queries_semantic_preds.shape[-1]),
                center_queries_semantic_labels.reshape(-1)
            )
            loss_out['center_semantic_loss'] = (center_queries_semantic_loss, center_queries_semantic_labels.shape[0])

            queries_offset_preds, center_queries_offsets = loss_inp['queries_offset_preds']
            center_queries_offset_loss = center_offset_criterion(queries_offset_preds, center_queries_offsets)
            loss_out['center_offset_loss'] = (center_queries_offset_loss, center_queries_offsets.shape[0])

        if 'points_semantic_center_loss_feature' in loss_inp.keys():
            points_semantic_center_loss_features, labels, batch_offsets = loss_inp['points_semantic_center_loss_feature']

            semantic_center_loss = torch.zeros(1).cuda()
            for points_semantic_center_loss_feature in points_semantic_center_loss_features:
                semantic_cent_loss = torch.zeros(1).cuda()
                for batch_id in range(1, len(batch_offsets)):
                    points_semantic_feat = points_semantic_center_loss_feature[
                                           batch_offsets[batch_id - 1]:batch_offsets[batch_id]]
                    label = labels[batch_offsets[batch_id - 1]:batch_offsets[batch_id]]
                    semantic_cent_loss += semantic_center_criterion(points_semantic_feat, label)
                semantic_center_loss += semantic_cent_loss / (len(batch_offsets) - 1)

            loss_out['center_offset_loss'] = (semantic_center_loss, labels.shape[0])

        if cfg.instance_triplet_loss['activate'] and 'point_offset_feats' in loss_inp.keys():
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
                                                    cfg.instance_triplet_loss['num_sampled_points']))))

                triplet_index = torch.cat(triplet_index)

                instance_triplet_loss += instance_triplet_criterion(
                    point_offset_feat[triplet_index], instance_label[triplet_index])[0]

            instance_triplet_loss = instance_triplet_loss / (len(batch_offsets) - 1)

            loss_out['center_offset_loss'] = (instance_triplet_loss, instance_labels.shape[0])

        ### three different feature term losses mentioned in OccuSeg
        if cfg.feature_variance_loss['activate'] and ('feature_variance_loss' in loss_inp.keys()):
            point_features, instance_labels = loss_inp['feature_variance_loss']

            valid_instance_index = (instance_labels != cfg.ignore_label)
            instance_features = scatter_mean(
                point_features[valid_instance_index], instance_labels[valid_instance_index], dim=0
            )
            feature_variance_loss = scatter_mean(
                torch.relu(
                    torch.norm(
                        instance_features[instance_labels[valid_instance_index]] - \
                        point_features[valid_instance_index], p=2, dim=1
                    ) - cfg.feature_variance_loss['variance_threshold']) ** 2,
                instance_labels[valid_instance_index], dim=0
            ).mean()

            loss_out['feature_variance_loss'] = (feature_variance_loss, instance_labels.shape[0])

        if cfg.feature_distance_loss['activate'] and ('feature_distance_loss' in loss_inp.keys()):
            point_features, instance_labels = loss_inp['feature_distance_loss']

            valid_instance_index = (instance_labels != cfg.ignore_label)
            instance_features = scatter_mean(
                point_features[valid_instance_index], instance_labels[valid_instance_index], dim=0
            )
            instance_dist_mat = torch.norm(
                instance_features.unsqueeze(dim=0) - instance_features.unsqueeze(dim=1), dim=2)
            instance_dist_mat = torch.relu(
                (2 * cfg.feature_distance_loss['distance_threshold'] - instance_dist_mat) ** 2)
            instance_dist_mat[range(len(instance_dist_mat)), range(len(instance_dist_mat))] = 0
            feature_distance_loss = instance_dist_mat.sum() / (
                        instance_dist_mat.shape[0] * (instance_dist_mat.shape[0] - 1))

            loss_out['feature_distance_loss'] = (feature_distance_loss, instance_labels.shape[0])

        if cfg.feature_instance_regression_loss['activate'] and ('feature_instance_regression_loss' in loss_inp.keys()):
            point_features, instance_labels = loss_inp['feature_instance_regression_loss']

            valid_instance_index = (instance_labels != cfg.ignore_label)
            instance_features = scatter_mean(
                point_features[valid_instance_index], instance_labels[valid_instance_index], dim=0
            )
            feature_instance_regression_loss = torch.mean(torch.norm(instance_features, p=2, dim=1), dim=0)

            loss_out['feature_instance_regression_loss'] = (feature_instance_regression_loss, instance_labels.shape[0])

        if cfg.feature_semantic_regression_loss['activate'] and ('feature_semantic_regression_loss' in loss_inp.keys()):
            point_features, semantic_labels = loss_inp['feature_semantic_regression_loss']

            valid_semantic_index = (semantic_labels != cfg.ignore_label)
            semantic_features = scatter_mean(
                point_features[valid_semantic_index], semantic_labels[valid_semantic_index], dim=0
            )
            feature_semantic_regression_loss = torch.norm(torch.sum(semantic_features, dim=0), p=2) / \
                                               semantic_features.shape[0]

            loss_out['feature_semantic_regression_loss'] = (feature_semantic_regression_loss, semantic_labels.shape[0])

        ### occupancy loss to predict the voxel number for each voxel
        if ('occupancy' in cfg.model_mode.split('_')) and ('voxel_occupancy_loss' in loss_inp.keys()):
            point_occupancy_preds, voxel_instance_labels, voxel_occupancy_labels = loss_inp['voxel_occupancy_loss']

            voxel_occupancy_loss = torch.zeros(1).cuda()
            for point_occupancy_pred in point_occupancy_preds:
                valid_voxel_index = voxel_instance_labels != -100
                point_occupancy_prediction = point_occupancy_pred[valid_voxel_index].squeeze(dim=1)
                voxel_instance_labels = voxel_instance_labels[valid_voxel_index]
                voxel_occupancy_labels = voxel_occupancy_labels[valid_voxel_index]
                voxel_occupancy_loss += scatter_mean(
                    torch.abs(point_occupancy_prediction - torch.log(voxel_occupancy_labels.float())),
                    voxel_instance_labels
                ).mean()

            voxel_occupancy_loss = voxel_occupancy_loss / len(point_occupancy_preds)
            loss_out['voxel_occupancy_loss'] = (voxel_occupancy_loss, point_occupancy_preds[0].shape[0])

        if ('local_point_semantic' in loss_inp.keys()) and (cfg.local_proposal['scatter_mean_target'] == False):
            # local proposal point semantic loss calculate
            local_point_semantic_scores, local_proposals_idx, labels = loss_inp['local_point_semantic']

            local_point_semantic_loss = torch.zeros(1).cuda()
            for local_point_semantic_score in local_point_semantic_scores:
                local_point_semantic_loss += semantic_criterion(local_point_semantic_score, labels[local_proposals_idx[:, 1].long()])

            loss_out['local_point_semantic_loss'] = (local_point_semantic_loss, local_proposals_idx.shape[0])

            # local proposal point offset losses
            local_point_offset_preds, local_proposals_idx, coords, instance_info, instance_labels = loss_inp['local_point_offset']

            local_point_offset_norm_loss = torch.zeros(1).cuda()
            local_point_offset_dir_loss = torch.zeros(1).cuda()
            for local_point_offset_pred in local_point_offset_preds:
                gt_offsets = instance_info[:, 0:3] - coords  # (N, 3)
                gt_offsets = gt_offsets[local_proposals_idx[:, 1].long(), :]
                pt_diff = local_point_offset_pred - gt_offsets  # (N, 3)
                pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
                valid = (instance_labels[local_proposals_idx[:, 1].long()] != cfg.ignore_label).float()
                local_point_offset_norm_loss += torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

                gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
                gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
                local_point_offsets_norm = torch.norm(local_point_offset_pred, p=2, dim=1)
                local_point_offsets_ = local_point_offset_pred / (local_point_offsets_norm.unsqueeze(-1) + 1e-8)
                direction_diff = - (gt_offsets_ * local_point_offsets_).sum(-1)  # (N)
                local_point_offset_dir_loss += torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

            loss_out['local_point_offset_norm_loss'] = (local_point_offset_norm_loss, valid.sum())
            loss_out['local_point_offset_dir_loss'] = (local_point_offset_dir_loss, valid.sum())

        if ('point_reconstructed_coords' in loss_inp.keys()) and (cfg.point_xyz_reconstruction_loss['activate']):
            point_reconstructed_coords, coords_float = loss_inp['point_reconstructed_coords']

            point_xyz_reconstruction_loss = point_reconstruction_criterion(point_reconstructed_coords, coords_float)

            loss_out['point_xyz_reconstruction_loss'] = (point_xyz_reconstruction_loss, coords_float.shape[0])

        if ('point_reconstructed_colors' in loss_inp.keys()) and (cfg.point_rgb_reconstruction_loss['activate']):
            point_reconstructed_colors, rgb = loss_inp['point_reconstructed_colors']

            point_rgb_reconstruction_loss = point_reconstruction_criterion(point_reconstructed_colors, rgb)

            loss_out['point_rgb_reconstruction_loss'] = (point_rgb_reconstruction_loss, rgb.shape[0])

        if ('instance_id_preds' in loss_inp.keys()) and (cfg.instance_classifier['activate']):
            instance_id_preds, instance_labels = loss_inp['instance_id_preds']

            point_instance_id_loss = point_instance_id_criterion(instance_id_preds, instance_labels)

            loss_out['point_instance_id_loss'] = (point_instance_id_loss, instance_labels.shape[0])

        if ('local_point_feature_discriminative_loss' in loss_inp.keys()) and (cfg.local_proposal['local_point_feature_discriminative_loss']):
            proposals_point_features, proposals_idx, instance_labels = loss_inp['local_point_feature_discriminative_loss']

            local_feature_variance_loss = torch.zeros(1).cuda()
            local_feature_distance_loss = torch.zeros(1).cuda()
            local_feature_instance_regression_loss = torch.zeros(1).cuda()

            for proposal_idx in proposals_idx[:, 0].unique():
                proposal_valid_idx = (proposals_idx[:, 0] == proposal_idx)
                proposals_point_feature = proposals_point_features[proposal_valid_idx]
                proposal_valid_idx = proposals_idx[proposal_valid_idx, 1].long()
                instance_label = instance_labels[proposal_valid_idx]

                ## variance_loss
                valid_instance_index = (instance_label != cfg.ignore_label)
                instance_features = scatter_mean(
                    proposals_point_feature[valid_instance_index], instance_label[valid_instance_index], dim=0
                )
                local_feature_variance_loss += scatter_mean(
                    torch.relu(
                        torch.norm(
                            instance_features[instance_label[valid_instance_index]] - \
                            proposals_point_feature[valid_instance_index], p=2, dim=1
                        ) - cfg.feature_variance_loss['variance_threshold']) ** 2,
                    instance_label[valid_instance_index], dim=0
                ).mean()

                ## distance_loss
                instance_features = scatter_mean(
                    proposals_point_feature[valid_instance_index], instance_label[valid_instance_index], dim=0
                )
                instance_dist_mat = torch.norm(
                    instance_features.unsqueeze(dim=0) - instance_features.unsqueeze(dim=1), dim=2)
                instance_dist_mat = torch.relu(
                    (2 * cfg.feature_distance_loss['distance_threshold'] - instance_dist_mat) ** 2)
                instance_dist_mat[range(len(instance_dist_mat)), range(len(instance_dist_mat))] = 0
                local_feature_distance_loss += instance_dist_mat.sum() / (
                            instance_dist_mat.shape[0] * (instance_dist_mat.shape[0] - 1))

                ## instance_regression_loss
                instance_features = scatter_mean(
                    proposals_point_feature[valid_instance_index], instance_label[valid_instance_index], dim=0
                )
                local_feature_instance_regression_loss += torch.mean(torch.norm(instance_features, p=2, dim=1), dim=0)

            local_feature_variance_loss = local_feature_variance_loss / len(proposals_idx[:, 0].unique())
            local_feature_distance_loss = local_feature_distance_loss / len(proposals_idx[:, 0].unique())
            local_feature_instance_regression_loss = local_feature_instance_regression_loss / len(proposals_idx[:, 0].unique())

        ### ============================================================================================================
        '''total loss'''
        loss = cfg.loss_weights['point_semantic'] * semantic_loss + \
               cfg.loss_weights['point_offset_norm'] * offset_norm_loss + \
               cfg.loss_weights['point_offset_dir'] * offset_dir_loss

        if ('center_preds' in loss_inp.keys()):
            if len(loss_inp['center_preds']) == 2:
                loss += cfg.loss_weights['center_prob'] * center_loss + \
                        cfg.loss_weights['center_semantic'] * center_semantic_loss + \
                        cfg.loss_weights['center_offset'] * center_offset_loss
            elif len(loss_inp['center_preds']) == 6:
                loss += cfg.loss_weights['center_prob'] * center_probs_loss + \
                        cfg.loss_weights['center_semantic'] * center_semantic_loss + \
                        cfg.loss_weights['center_offset_norm_loss'] * center_offset_norm_loss + \
                        cfg.loss_weights['center_offset_dir_loss'] * center_offset_dir_loss

        if (epoch > cfg.prepare_epochs) and ('proposal_scores' in loss_inp.keys()):
            loss += cfg.loss_weights['score'] * score_loss

        if ('proposal_confidences' in loss_inp.keys()):
            loss += confidence_loss

        if ('stuff_preds' in loss_inp.keys()):
            loss += stuff_loss

        if cfg.stuff_norm_loss['activate'] and ('output_feats' in loss_inp.keys()):
            loss += cfg.loss_weights['stuff_feats_norm'] * stuff_feats_norm_loss

        if ('queries_preds' in loss_inp.keys()):
            loss = loss + center_queries_loss + center_queries_semantic_loss + center_queries_offset_loss

        if ('points_semantic_center_loss_feature' in loss_inp.keys()):
            loss += cfg.loss_weights['point_semantic_center'] * semantic_center_loss

        if cfg.instance_triplet_loss['activate'] and ('point_offset_feats' in loss_inp.keys()):
            loss += cfg.loss_weights['point_instance_triplet'] * instance_triplet_loss

        if cfg.feature_instance_regression_loss['activate'] and ('feature_variance_loss' in loss_inp.keys()):
            loss += cfg.loss_weights['feature_variance_loss'] * feature_variance_loss

        if cfg.feature_instance_regression_loss['activate'] and ('feature_distance_loss' in loss_inp.keys()):
            loss += cfg.loss_weights['feature_distance_loss'] * feature_distance_loss

        if cfg.feature_instance_regression_loss['activate'] and ('feature_instance_regression_loss' in loss_inp.keys()):
            loss += cfg.loss_weights['feature_instance_regression_loss'] * feature_instance_regression_loss

        if cfg.feature_semantic_regression_loss['activate'] and ('feature_semantic_regression_loss' in loss_inp.keys()):
            loss += cfg.loss_weights['feature_semantic_regression_loss'] * feature_semantic_regression_loss

        if ('occupancy' in cfg.model_mode.split('_')) and ('voxel_occupancy_loss' in loss_inp.keys()):
            loss += cfg.loss_weights['voxel_occupancy_loss'] * voxel_occupancy_loss

        if ('local_point_semantic' in loss_inp.keys()) and (cfg.local_proposal['scatter_mean_target'] == False):
            loss += cfg.loss_weights['local_point_semantic_loss'] * local_point_semantic_loss + \
                    cfg.loss_weights['local_point_offset_norm'] * local_point_offset_norm_loss + \
                    cfg.loss_weights['local_point_offset_dir'] * local_point_offset_dir_loss

        if ('point_reconstructed_coords' in loss_inp.keys()) and (cfg.point_xyz_reconstruction_loss['activate']):
            loss += cfg.loss_weights['point_xyz_reconstruction_loss'] * point_xyz_reconstruction_loss

        if ('point_reconstructed_colors' in loss_inp.keys()) and (cfg.point_rgb_reconstruction_loss['activate']):
            loss += cfg.loss_weights['point_rgb_reconstruction_loss'] * point_rgb_reconstruction_loss

        if ('instance_id_preds' in loss_inp.keys()) and (cfg.instance_classifier['activate']):
            loss += cfg.loss_weights['point_instance_id_loss'] * point_instance_id_loss

        if ('local_point_feature_discriminative_loss' in loss_inp.keys()) and (cfg.local_proposal['local_point_feature_discriminative_loss']):
            loss += cfg.loss_weights['local_feature_variance_loss'] * local_feature_variance_loss + \
                    cfg.loss_weights['local_feature_distance_loss'] * local_feature_distance_loss + \
                    cfg.loss_weights['local_feature_instance_regression_loss'] * local_feature_instance_regression_loss

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
