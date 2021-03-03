import torch


def compute_offset_norm_loss(offset_preds, offset_gts, instance_labels, ignore_label=-100):

    pt_diff = offset_preds - offset_gts  # (N, 3)
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
    valid = (instance_labels != ignore_label).float()
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

    return offset_norm_loss


def compute_offset_dir_loss(offset_preds, offset_gts, instance_labels, ignore_label=-100):

    gt_offsets_norm = torch.norm(offset_gts, p=2, dim=1)  # (N), float
    gt_offsets_ = offset_gts / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
    pt_offsets_norm = torch.norm(offset_preds, p=2, dim=1)
    pt_offsets_ = offset_preds / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
    direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
    valid = (instance_labels != ignore_label).float()
    offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

    return offset_dir_loss


