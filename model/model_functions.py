import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_scatter import scatter_mean
import sys

sys.path.append('../')

from lib.pointgroup_ops.functions import pointgroup_ops
from model.components import WeightedFocalLoss
from model import loss_functions


def model_fn_decorator(cfg, test=False):
    '''criterions'''
    criterions = {}

    criterions['point_semantic_criterion'] = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    if cfg.offset_norm_criterion == 'l2':
        criterions['point_offset_norm_criterion'] = nn.MSELoss().cuda()

    criterions['center_prob_criterion'] = WeightedFocalLoss(
        alpha=cfg.focal_loss['alpha'], gamma=cfg.focal_loss['gamma']).cuda()
    criterions['center_semantic_criterion'] = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    criterions['score_criterion'] = nn.BCELoss(reduction='none').cuda()
    criterions['confidence_criterion'] = nn.BCELoss(reduction='none').cuda()

    criterions['point_xyz_reconstruction_criterion'] = nn.SmoothL1Loss().cuda()
    criterions['point_instance_id_criterion'] = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

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

        if ('occupancy' in cfg.model_mode.split('_')):
            voxel_instance_labels = batch['voxel_instance_labels'].cuda()
            voxel_occupancy_labels = batch['voxel_occupancy_labels'].cuda()

        ret = model(
            input_, p2v_map, coords_float, rgb, ori_coords,
            coords[:, 0].int(), batch_offsets, epoch
        )

        loss_inp = {}

        if ('point_offset_preds' in ret.keys()):
            point_semantic_scores = ret['point_semantic_scores']
            point_offset_preds = ret['point_offset_preds']

            loss_inp['point_semantic_loss'] = (point_semantic_scores, labels)
            loss_inp['point_offset_loss'] = (
                point_offset_preds, coords_float, instance_info, instance_centers, instance_labels
            )

        if 'center_preds' in ret.keys():
            center_preds, sampled_indexes = ret['center_preds'] # (B, 8196, 1)
            center_semantic_preds, _ = ret['center_semantic_preds'] # (B, 8196, 20)
            center_offset_preds, _ = ret['center_offset_preds'] # (B, 8196, 3)

            loss_inp['center_prob_loss'] = (
                center_preds, coords_float, sampled_indexes, instance_centers, instance_sizes, batch_offsets
            )
            loss_inp['center_semantic_loss'] = (center_semantic_preds, sampled_indexes, labels, batch_offsets)
            loss_inp['center_offset_loss'] = (
                center_offset_preds, sampled_indexes, coords_float, instance_info, instance_labels, batch_offsets
            )

        if (epoch > cfg.prepare_epochs) and ('proposal_scores' in ret.keys()):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu

            loss_inp['proposal_score_loss'] = (
                scores, proposals_idx, proposals_offset, instance_labels, instance_pointnum)

        if 'proposal_confidences' in ret.keys():
            proposals_confidence_preds, proposals_idx_shifts, proposals_offset_shifts = ret['proposal_confidences']
            loss_inp['proposal_confidence_loss'] = (
                proposals_confidence_preds, proposals_idx_shifts,
                proposals_offset_shifts, instance_labels, instance_pointnum
            )

        if 'stuff_preds' in ret.keys():
            stuff_preds = ret['stuff_preds']
            loss_inp['stuff_preds'] = (stuff_preds, labels)

        if 'points_semantic_center_loss_feature' in ret.keys():
            points_semantic_center_loss_feature = ret['points_semantic_center_loss_feature']

            loss_inp['points_semantic_center_loss_feature'] = (points_semantic_center_loss_feature, labels, batch_offsets)

        if cfg.instance_triplet_loss['activate'] and 'point_offset_feats' in ret.keys():
            point_offset_feats = ret['point_offset_feats']

            loss_inp['point_offset_feats'] = (point_offset_feats, instance_labels, batch_offsets)

        ### try three different feature term losses mentioned in OccuSeg
        if cfg.feature_instance_variance_loss['activate'] and ('point_features' in ret.keys()):
            point_features = ret['point_features']
            loss_inp['feature_instance_variance_loss'] = (point_features, instance_labels)

        if cfg.feature_instance_distance_loss['activate'] and ('point_features' in ret.keys()):
            point_features = ret['point_features']
            loss_inp['feature_instance_distance_loss'] = (point_features, instance_labels)

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

            loss_inp['local_point_semantic_loss'] = (local_point_semantic_scores, local_proposals_idx, labels)
            loss_inp['local_point_offset_loss'] = (
                local_point_offset_preds, local_proposals_idx, coords_float, instance_info, instance_labels)

        if ('point_reconstructed_coords' in ret.keys()) and (cfg.point_xyz_reconstruction_loss['activate']):
            point_reconstructed_coords = ret['point_reconstructed_coords']

            loss_inp['point_xyz_reconstruction_loss'] = (point_reconstructed_coords, coords_float)

        if ('instance_id_preds' in ret.keys()) and (cfg.instance_classifier['activate']):
            instance_id_preds = ret['instance_id_preds']

            loss_inp['point_instance_id_loss'] = (instance_id_preds, instance_labels)

        if ('proposals_point_features' in ret.keys()) and (cfg.local_proposal['local_point_feature_discriminative_loss']):
            proposals_point_features, proposals_idx = ret['proposals_point_features']

            loss_inp['local_point_feature_discriminative_loss'] = (proposals_point_features, proposals_idx, instance_labels)

        if ('voxel_center_preds' in ret.keys()) and (cfg.voxel_center_prediction['activate']):
            voxel_center_preds = ret['voxel_center_preds']
            voxel_center_offset_preds = ret['voxel_center_offset_preds']
            voxel_center_semantic_preds = ret['voxel_center_semantic_preds']

            voxel_center_probs_labels = batch['voxel_center_probs_labels'].cuda()
            loss_inp['voxel_center_prob_loss'] = (voxel_center_preds, voxel_center_probs_labels)
            voxel_center_semantic_labels = batch['voxel_center_semantic_labels'].cuda()
            loss_inp['voxel_center_semantic_loss'] = (voxel_center_semantic_preds, voxel_center_semantic_labels)
            voxel_center_offset_labels = batch['voxel_center_offset_labels'].cuda()
            voxel_center_instance_labels = batch['voxel_center_instance_labels'].cuda()
            loss_inp['voxel_center_offset_loss'] = (voxel_center_offset_preds, voxel_center_offset_labels, voxel_center_instance_labels)

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
        loss = torch.zeros(1).cuda()
        loss_out = {}
        infos = {}

        for loss_name in loss_inp:
            loss_func = getattr(loss_functions, loss_name)
            loss += loss_func(cfg, criterions, loss_inp[loss_name], loss_out)

        return loss, loss_out, infos

    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn
