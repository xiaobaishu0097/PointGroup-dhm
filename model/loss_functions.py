import torch

from torch_scatter import scatter_mean

from model.common import generate_adaptive_heatmap
from lib.pointgroup_ops.functions import pointgroup_ops


def compute_offset_norm_loss(offset_preds, offset_gts, instance_labels, ignore_label=-100):
    """

    Args:
        offset_preds:
        offset_gts:
        instance_labels:
        ignore_label:

    Returns:

    """
    pt_diff = offset_preds - offset_gts  # (N, 3)
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
    valid = (instance_labels != ignore_label).float()
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

    return offset_norm_loss


def compute_offset_dir_loss(offset_preds, offset_gts, instance_labels, ignore_label=-100):
    """

    Args:
        offset_preds:
        offset_gts:
        instance_labels:
        ignore_label:

    Returns:

    """
    gt_offsets_norm = torch.norm(offset_gts, p=2, dim=1)  # (N), float
    gt_offsets_ = offset_gts / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
    pt_offsets_norm = torch.norm(offset_preds, p=2, dim=1)
    pt_offsets_ = offset_preds / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
    direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
    valid = (instance_labels != ignore_label).float()
    offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

    return offset_dir_loss


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


def point_semantic_loss(cfg, criterions, loss_inp, loss_out):
    """compute point-wise semantic loss

    Args:
        cfg:
        criterions:
        loss_inp: (semantic_scores: (N, nClass), float32, cuda, semantic_labels: (N), long, cuda)
        loss_out:

    """
    semantic_scores, semantic_labels = loss_inp

    point_semantic_loss = torch.zeros(1).cuda(semantic_scores[0].device)
    for semantic_score in semantic_scores:
        point_semantic_loss += criterions['point_semantic_criterion'](semantic_score, semantic_labels.to(semantic_score.device))

    loss_out['point_semantic_loss'] = (point_semantic_loss, semantic_scores[0].shape[0])

    loss = cfg.loss_weights['point_semantic'] * point_semantic_loss

    return loss


def point_offset_loss(cfg, criterions, loss_inp, loss_out):
    pt_offsets, coords, instance_info, instance_center, instance_labels = loss_inp
    # pt_offsets: (N, 3), float, cuda
    # coords: (N, 3), float32
    # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
    # instance_labels: (N), long

    point_offset_norm_loss = torch.zeros(1).cuda()
    point_offset_dir_loss = torch.zeros(1).cuda()
    for pt_offset in pt_offsets:
        gt_offset = instance_info[:, 0:3] - coords  # (N, 3)

        '''point offset norm loss'''
        if cfg.offset_norm_criterion == 'l1':
            point_offset_norm_loss += compute_offset_norm_loss(
                pt_offset, gt_offset, instance_labels, ignore_label=cfg.ignore_label)
        if cfg.offset_norm_criterion == 'l2':
            pt_valid_index = instance_labels != cfg.ignore_label
            point_offset_norm_loss += criterions['offset_norm_criterion'](pt_offset[pt_valid_index], gt_offset[pt_valid_index])

        '''point offset dir loss'''
        point_offset_dir_loss += compute_offset_dir_loss(
            pt_offset, gt_offset, instance_labels, ignore_label=cfg.ignore_label)

    loss_out['point_offset_norm_loss'] = (point_offset_norm_loss, (instance_labels != cfg.ignore_label).sum())
    loss_out['point_offset_dir_loss'] = (point_offset_dir_loss, (instance_labels != cfg.ignore_label).sum())

    loss = cfg.loss_weights['point_offset_norm'] * point_offset_norm_loss + \
           cfg.loss_weights['point_offset_dir'] * point_offset_dir_loss

    return loss


def center_prob_loss(cfg, criterions, loss_inp, loss_out):
    center_preds, point_coords, sampled_indexes, instance_centers, instance_sizes, batch_offsets = loss_inp

    ### compute center probability ground truth heatmap
    center_heatmaps = []
    for batch_index in range(1, len(batch_offsets)):
        point_coord = point_coords[batch_offsets[batch_index - 1]:batch_offsets[batch_index]]
        instance_center = instance_centers[instance_centers[:, 0] == (batch_index - 1), 1:]
        instance_size = instance_sizes[instance_centers[:, 0] == (batch_index - 1), 1:]

        ### produce center probability of sampled points
        sampled_index = sampled_indexes[sampled_indexes[:, 0] == batch_index, 1]
        center_heatmap = generate_adaptive_heatmap(
            point_coord[sampled_index, :].double().cpu(), instance_center.cpu(),
            instance_size.cpu(), min_IoU=cfg.min_IoU
        )['heatmap']
        center_heatmaps.append(center_heatmap.cuda())

    '''center prob loss'''
    center_heatmaps = torch.cat(center_heatmaps, dim=0).to(torch.float32)
    center_prob_loss = criterions['center_prob_criterion'](center_preds.view(-1), center_heatmaps)

    loss_out['center_probs_loss'] = (center_prob_loss, sampled_indexes.shape[0])

    loss = cfg.loss_weights['center_prob_loss'] * center_prob_loss

    return loss


def center_semantic_loss(cfg, criterions, loss_inp, loss_out):
    center_semantic_preds, sampled_indexes, point_semantic_labels, batch_offsets = loss_inp

    center_semantic_labels = []
    for batch_index in range(1, len(batch_offsets)):
        sampled_index = sampled_indexes[sampled_indexes[:, 0] == batch_index, 1]
        point_semantic_label = point_semantic_labels[batch_offsets[batch_index - 1]:batch_offsets[batch_index]]
        center_semantic_labels.append(point_semantic_label[sampled_index])

    '''center semantic loss'''
    center_semantic_labels = torch.cat(center_semantic_labels, dim=0)
    center_semantic_loss = criterions['center_semantic_criterion'](
        center_semantic_preds.view(-1, 20), center_semantic_labels
    )

    loss_out['center_semantic_loss'] = (center_semantic_loss, sampled_indexes.shape[0])

    loss = cfg.loss_weights['center_semantic_loss'] * center_semantic_loss

    return loss


def center_offset_loss(cfg, criterions, loss_inp, loss_out):
    center_offset_preds, sampled_indexes, point_coords, point_instance_info, instance_labels, batch_offsets = loss_inp

    '''center offset loss'''
    center_offset_norm_loss = torch.zeros(1).cuda()
    center_offset_dir_loss = torch.zeros(1).cuda()
    for batch_index in range(1, len(batch_offsets)):
        sampled_index = sampled_indexes[sampled_indexes[:, 0] == batch_index, 1]

        ### select center information based on selected sample indexes
        center_instance_info = point_instance_info[batch_offsets[batch_index - 1]:batch_offsets[batch_index]]
        center_instance_info = center_instance_info[sampled_index]
        center_coord = point_coords[batch_offsets[batch_index - 1]:batch_offsets[batch_index]]
        center_coord = center_coord[sampled_index]
        center_offset_preds = center_offset_preds.view(-1, 3)
        center_offset_pred = center_offset_preds[sampled_indexes[:, 0] == batch_index, :]
        instance_label = instance_labels[batch_offsets[batch_index - 1]:batch_offsets[batch_index]]
        instance_label = instance_label[sampled_index]
        center_gt_offsets = center_instance_info[:, 0:3] - center_coord

        center_offset_norm_loss += compute_offset_norm_loss(
            center_offset_pred, center_gt_offsets, instance_label, ignore_label=cfg.ignore_label)
        center_offset_dir_loss += compute_offset_dir_loss(
            center_offset_pred, center_gt_offsets, instance_label, ignore_label=cfg.ignore_label)

    loss_out['center_offset_norm_loss'] = (center_offset_norm_loss, sampled_indexes.shape[0])
    loss_out['center_offset_dir_loss'] = (center_offset_dir_loss, sampled_indexes.shape[0])

    loss = cfg.loss_weights['center_offset_norm_loss'] * center_offset_norm_loss + \
           cfg.loss_weights['center_offset_dir_loss'] * center_offset_dir_loss

    return loss


def proposal_score_loss(cfg, criterions, loss_inp, loss_out):
    scores, proposals_idx, proposals_offset, instance_labels, instance_pointnum = loss_inp
    # scores: (nProposal, 1), float32
    # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
    # proposals_offset: (nProposal + 1), int, cpu
    # instance_pointnum: (total_nInst), int

    '''proposal score loss'''
    ious = pointgroup_ops.get_iou(
        proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels, instance_pointnum
    )  # (nProposal, nInstance), float
    gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
    gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

    proposal_score_loss = criterions['score_criterion'](torch.sigmoid(scores.view(-1)), gt_scores)
    proposal_score_loss = proposal_score_loss.mean()

    loss_out['proposal_score_loss'] = (proposal_score_loss, gt_ious.shape[0])

    loss = cfg.loss_weights['score'] * proposal_score_loss

    return loss


def proposal_confidence_loss(cfg, criterions, loss_inp, loss_out):
    proposals_confidence_preds, proposals_idx_shifts, proposals_offset_shifts, instance_labels, instance_pointnum = loss_inp
    # scores: (nProposal, 1), float32
    # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
    # proposals_offset: (nProposal + 1), int, cpu
    # instance_pointnum: (total_nInst), int

    proposal_confidence_loss = torch.zeros(1).cuda()

    for proposal_index in range(len(proposals_confidence_preds)):
        ious = pointgroup_ops.get_iou(
            proposals_idx_shifts[proposal_index][:, 1].cuda(), proposals_offset_shifts[proposal_index].cuda(),
            instance_labels, instance_pointnum
        )  # (nProposal, nInstance), float
        gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
        gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)

        conf_loss = criterions['confidence_criterion'](
            torch.sigmoid(proposals_confidence_preds[proposal_index].view(-1)), gt_scores)
        proposal_confidence_loss += conf_loss.mean()

    loss_out['proposal_confidence_loss'] = (proposal_confidence_loss, gt_ious.shape[0])

    loss = cfg.loss_weights['proposal_confidence_loss'] * proposal_confidence_loss

    return loss


def feature_instance_variance_loss(cfg, criterions, loss_inp, loss_out):
    point_features, instance_labels = loss_inp

    valid_instance_index = (instance_labels != cfg.ignore_label)
    instance_features = scatter_mean(
        point_features[valid_instance_index], instance_labels[valid_instance_index], dim=0
    )
    feature_instance_variance_loss = scatter_mean(
        torch.relu(
            torch.norm(
                instance_features[instance_labels[valid_instance_index]] - \
                point_features[valid_instance_index], p=2, dim=1
            ) - cfg.feature_variance_loss['variance_threshold']) ** 2,
        instance_labels[valid_instance_index], dim=0
    ).mean()

    loss_out['feature_variance_loss'] = (feature_instance_variance_loss, instance_labels.shape[0])

    loss = cfg.loss_weights['feature_instance_variance_loss'] * feature_instance_variance_loss

    return loss


def feature_instance_distance_loss(cfg, criterions, loss_inp, loss_out):
    point_features, instance_labels = loss_inp

    valid_instance_index = (instance_labels != cfg.ignore_label)
    instance_features = scatter_mean(
        point_features[valid_instance_index], instance_labels[valid_instance_index], dim=0
    )
    instance_dist_mat = torch.norm(
        instance_features.unsqueeze(dim=0) - instance_features.unsqueeze(dim=1), dim=2)
    instance_dist_mat = torch.relu(
        (2 * cfg.feature_distance_loss['distance_threshold'] - instance_dist_mat) ** 2)
    instance_dist_mat[range(len(instance_dist_mat)), range(len(instance_dist_mat))] = 0
    feature_instance_distance_loss = instance_dist_mat.sum() / (
                instance_dist_mat.shape[0] * (instance_dist_mat.shape[0] - 1))

    loss_out['feature_distance_loss'] = (feature_instance_distance_loss, instance_labels.shape[0])

    loss = cfg.loss_weights['feature_distance_loss'] * feature_instance_distance_loss

    return loss


def feature_instance_regression_loss(cfg, criterions, loss_inp, loss_out):
    point_features, instance_labels = loss_inp

    valid_instance_index = (instance_labels != cfg.ignore_label)
    instance_features = scatter_mean(
        point_features[valid_instance_index], instance_labels[valid_instance_index], dim=0
    )
    feature_instance_regression_loss = torch.mean(torch.norm(instance_features, p=2, dim=1), dim=0)

    loss_out['feature_instance_regression_loss'] = (feature_instance_regression_loss, instance_labels.shape[0])

    loss = cfg.loss_weights['feature_instance_regression_loss'] * feature_instance_regression_loss

    return loss


def voxel_occupancy_loss(cfg, criterions, loss_inp, loss_out):
    point_occupancy_preds, voxel_instance_labels, voxel_occupancy_labels = loss_inp

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

    loss = cfg.loss_weights['voxel_occupancy_loss'] * voxel_occupancy_loss

    return loss


def local_point_semantic_loss(cfg, criterions, loss_inp, loss_out):
    local_point_semantic_scores, local_proposals_idx, labels = loss_inp

    local_point_semantic_loss = torch.zeros(1).cuda()
    for local_point_semantic_score in local_point_semantic_scores:
        local_point_semantic_loss += criterions['point_semantic_criterion'](
            local_point_semantic_score, labels[local_proposals_idx[:, 1].long()])

    loss_out['local_point_semantic_loss'] = (local_point_semantic_loss, local_proposals_idx.shape[0])

    loss = cfg.loss_weights['local_point_semantic_loss'] * local_point_semantic_loss

    return loss


def local_point_offset_loss(cfg, criterions, loss_inp, loss_out):
    local_point_offset_preds, local_proposals_idx, coords, instance_info, instance_labels = loss_inp

    local_point_offset_norm_loss = torch.zeros(1).cuda()
    local_point_offset_dir_loss = torch.zeros(1).cuda()
    for local_point_offset_pred in local_point_offset_preds:
        gt_offsets = instance_info[:, 0:3] - coords  # (N, 3)
        local_point_gt_offsets = gt_offsets[local_proposals_idx[:, 1].long(), :]
        instance_label = instance_labels[local_proposals_idx[:, 1].long()]

        local_point_offset_norm_loss += compute_offset_norm_loss(
            local_point_offset_pred, local_point_gt_offsets, instance_label, ignore_label=cfg.ignore_label)
        local_point_offset_dir_loss += compute_offset_dir_loss(
            local_point_offset_pred, local_point_gt_offsets, instance_label, ignore_label=cfg.ignore_label)

    loss_out['local_point_offset_norm_loss'] = (local_point_offset_norm_loss, instance_label.shape[0])
    loss_out['local_point_offset_dir_loss'] = (local_point_offset_dir_loss, instance_label.shape[0])

    loss = cfg.loss_weights['local_point_offset_norm'] * local_point_offset_norm_loss + \
           cfg.loss_weights['local_point_offset_dir'] * local_point_offset_dir_loss

    return loss


def point_xyz_reconstruction_loss(cfg, criterions, loss_inp, loss_out):
    point_reconstructed_coords, coords_float = loss_inp

    point_xyz_reconstruction_loss = criterions['point_xyz_reconstruction_criterion'](
        point_reconstructed_coords, coords_float)

    loss_out['point_xyz_reconstruction_loss'] = (point_xyz_reconstruction_loss, coords_float.shape[0])

    loss = cfg.loss_weights['point_xyz_reconstruction_loss'] * point_xyz_reconstruction_loss

    return loss


def point_instance_id_loss(cfg, criterions, loss_inp, loss_out):
    instance_id_preds, instance_labels = loss_inp

    point_instance_id_loss = criterions['point_instance_id_criterion'](instance_id_preds, instance_labels)

    loss_out['point_instance_id_loss'] = (point_instance_id_loss, instance_labels.shape[0])

    loss = cfg.loss_weights['point_instance_id_loss'] * point_instance_id_loss

    return loss


def local_point_feature_discriminative_loss(cfg, criterions, loss_inp, loss_out):
    proposals_point_features, proposals_idx, instance_labels = loss_inp

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
                ) - cfg.feature_instance_variance_loss['variance_threshold']) ** 2,
            instance_label[valid_instance_index], dim=0
        ).mean()

        ## distance_loss
        instance_features = instance_features[instance_features != torch.zeros(32).cuda()]
        if instance_features.ndimension() == 1:
            instance_features = instance_features.unsqueeze(dim=0)
        instance_dist_mat = torch.norm(
            instance_features.unsqueeze(dim=0) - instance_features.unsqueeze(dim=1), dim=2)
        instance_dist_mat = torch.relu(
            (2 * cfg.feature_instance_distance_loss['distance_threshold'] - instance_dist_mat) ** 2)
        instance_dist_mat[range(len(instance_dist_mat)), range(len(instance_dist_mat))] = 0
        local_feature_distance_loss += instance_dist_mat.sum() / max(
            (instance_dist_mat.shape[0] * (instance_dist_mat.shape[0] - 1)), 1)

        ## instance_regression_loss
        local_feature_instance_regression_loss += torch.mean(torch.norm(instance_features, p=2, dim=1), dim=0)

    local_feature_variance_loss = local_feature_variance_loss / len(proposals_idx[:, 0].unique())
    local_feature_distance_loss = local_feature_distance_loss / len(proposals_idx[:, 0].unique())
    local_feature_instance_regression_loss = local_feature_instance_regression_loss / len(proposals_idx[:, 0].unique())

    loss_out['local_feature_variance_loss'] = (local_feature_variance_loss, len(proposals_idx))
    loss_out['local_feature_distance_loss'] = (local_feature_distance_loss, len(proposals_idx))
    loss_out['local_feature_instance_regression_loss'] = (local_feature_instance_regression_loss, len(proposals_idx))

    loss = cfg.loss_weights['local_feature_variance_loss'] * local_feature_variance_loss + \
           cfg.loss_weights['local_feature_distance_loss'] * local_feature_distance_loss + \
           cfg.loss_weights['local_feature_instance_regression_loss'] * local_feature_instance_regression_loss

    return loss


def voxel_center_prob_loss(cfg, criterions, loss_inp, loss_out):
    voxel_center_preds, voxel_center_probs_labels = loss_inp

    voxel_center_loss = criterions['center_prob_criterion'](voxel_center_preds.view(-1), voxel_center_probs_labels)

    loss_out['voxel_center_prob_loss'] = (voxel_center_loss, voxel_center_probs_labels.shape[0])

    loss = cfg.loss_weights['voxel_center_prob_loss'] * voxel_center_loss

    return loss


def voxel_center_semantic_loss(cfg, criterions, loss_inp, loss_out):
    voxel_center_semantic_preds, voxel_center_semantic_labels = loss_inp

    voxel_center_semantic_loss = criterions['center_semantic_criterion'](
        voxel_center_semantic_preds, voxel_center_semantic_labels)

    loss_out['voxel_center_semantic_loss'] = (voxel_center_semantic_loss, voxel_center_semantic_labels.shape[0])

    loss = cfg.loss_weights['voxel_center_semantic_loss'] * voxel_center_semantic_loss

    return loss


def voxel_center_offset_loss(cfg, criterions, loss_inp, loss_out):

    voxel_center_offset_preds, voxel_center_offset_labels, voxel_center_instance_labels = loss_inp

    voxel_center_offset_norm_loss = compute_offset_norm_loss(
        voxel_center_offset_preds, voxel_center_offset_labels,
        voxel_center_instance_labels, ignore_label=cfg.ignore_label
    )
    voxel_center_offset_dir_loss = compute_offset_dir_loss(
        voxel_center_offset_preds, voxel_center_offset_labels,
        voxel_center_instance_labels, ignore_label=cfg.ignore_label
    )

    loss_out['voxel_center_offset_norm_loss'] = (voxel_center_offset_norm_loss, voxel_center_instance_labels.shape[0])
    loss_out['voxel_center_offset_dir_loss'] = (voxel_center_offset_dir_loss, voxel_center_instance_labels.shape[0])

    loss = cfg.loss_weights['voxel_center_offset_dir_loss'] * voxel_center_offset_dir_loss + \
           cfg.loss_weights['voxel_center_offset_norm_loss'] * voxel_center_offset_norm_loss

    return loss
