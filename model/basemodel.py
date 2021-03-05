import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import spconv
import functools
import os

from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
from model.common import coordinate2index, normalize_3d_coordinate

from model.components import ResidualBlock, VGGBlock


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.input_c = cfg.input_channel
        self.classes = cfg.classes
        self.block_reps = cfg.block_reps
        self.block_residual = cfg.block_residual

        self.cluster_radius = cfg.cluster_radius
        self.cluster_meanActive = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster_npoint_thre

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs
        self.test_epoch = cfg.test_epoch

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        self.point_based_backbone = cfg.point_based_backbone
        self.model_mode = cfg.model_mode
        self.reso_grid = 32
        self.cluster_sets = cfg.cluster_sets
        self.m = cfg.m

        self.pointnet_include_rgb = cfg.pointnet_include_rgb
        self.proposal_refinement = cfg.proposal_refinement
        self.point_xyz_reconstruction_loss = cfg.point_xyz_reconstruction_loss

        self.pointnet_max_npoint = 8196

        self.full_scale = cfg.full_scale
        self.batch_size = cfg.batch_size
        self.instance_triplet_loss = cfg.instance_triplet_loss

        self.instance_classifier = cfg.instance_classifier
        self.voxel_center_prediction = cfg.voxel_center_prediction
        self.local_rank = cfg.local_rank

        self.norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if self.block_residual:
            self.block = ResidualBlock
        else:
            self.block = VGGBlock

        if cfg.use_coords == True:
            self.input_c += 3
        elif cfg.use_coords == 'Z':
            self.input_c += 1

        self.unet3d = None

        self.module_map = {}

        # if self.pretrain_path is not None:
        #     map_location = {'cuda:0': 'cuda:{}'.format(cfg.local_rank)} if cfg.local_rank > 0 else None
        #     pretrain_dict = torch.load(self.pretrain_path, map_location=map_location)
        #     if 'module.' in list(pretrain_dict.keys())[0]:
        #         pretrain_dict = {k[len('module.'):]: v for k, v in pretrain_dict.items()}
        #     for m in self.pretrain_module:
        #         n1, n2 = utils.load_model_param(self.module_map[m], pretrain_dict, prefix=m)
        #         if cfg.local_rank == 0:
        #             print("[PID {}] Load pretrained ".format(os.getpid()) + m + ": {}/{}".format(n1, n2))
        #
        # #### fix parameter
        # for m in self.fix_module:
        #     mod = self.module_map[m]
        #     for param in mod.parameters():
        #         param.requires_grad = False

    def local_pretrained_model_parameter(self):
        if self.pretrain_path is not None:
            map_location = {'cuda:0': 'cuda:{}'.format(self.local_rank)} if self.local_rank > 0 else None
            pretrain_dict = torch.load(self.pretrain_path, map_location=map_location)
            if 'module.' in list(pretrain_dict.keys())[0]:
                pretrain_dict = {k[len('module.'):]: v for k, v in pretrain_dict.items()}
            for m in self.pretrain_module:
                n1, n2 = utils.load_model_param(self.module_map[m], pretrain_dict, prefix=m)
                if self.local_rank == 0:
                    print("[PID {}] Load pretrained ".format(os.getpid()) + m + ": {}/{}".format(n1, n2))

        #### fix parameter
        for m in self.fix_module:
            mod = self.module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = pointgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean, 0,
                                                  clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[
            0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(
            fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1,
                                                                       mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=0.1)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.m, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.m, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pointnet_backbone_forward(self, coords, ori_coords, rgb, batch_offsets):
        point_feats = []
        grid_feats = []

        if self.point_based_backbone == 'pointnet':
            for sample_indx in range(1, len(batch_offsets)):
                coords_input = coords[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)
                rgb_input = rgb[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)

                encoded_feats = self.pointnet_encoder(coords_input, rgb_input)
                '''encoded_feats:{
                coord: coordinate of points (b, n, 3)
                index: index of points (b, 1, n)
                point: feature of points (b, n, 32)
                grid: grid features (b, c, grid_dim, grid_dim, grid_dim)
                }
                '''

                point_feats.append(encoded_feats['point'].squeeze(dim=0))
                if self.unet3d is not None:
                    grid_feats.append(self.generate_grid_features(coords_input, encoded_feats['point']))

        elif self.point_based_backbone == 'pointnet++_yanx':
            for sample_indx in range(1, len(batch_offsets)):
                coords_input = coords[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)
                rgb_input = rgb[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)

                if self.pointnet_include_rgb:
                    pointnet_input = torch.cat((coords_input, rgb_input), dim=-1)
                else:
                    pointnet_input = coords_input
                _, point_feat = self.pointnet_encoder(pointnet_input)

                point_feats.append(point_feat.contiguous().squeeze(dim=0))
                if self.unet3d is not None:
                    grid_feats.append(self.generate_grid_features(coords_input, point_feat))

        elif self.point_based_backbone == 'pointnet++_shi':
            for sample_indx in range(1, len(batch_offsets)):
                coords_input = coords[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)
                rgb_input = rgb[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)
                ori_coords_input = ori_coords[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)

                point_feat, _ = self.pointnet_encoder(
                    coords_input, torch.cat((rgb_input, ori_coords_input), dim=2).transpose(1, 2).contiguous()
                )

                point_feats.append(point_feat.contiguous().squeeze(dim=0))
                if self.unet3d is not None:
                    grid_feats.append(self.generate_grid_features(coords_input, point_feat))

        point_feats = torch.cat(point_feats, 0).contiguous()
        if self.unet3d is not None:
            grid_feats = torch.cat(grid_feats, 0)

        return point_feats, grid_feats

    def pointgroup_cluster_algorithm(self, coords, point_offset_preds, point_semantic_preds, batch_idxs, batch_size, stuff_preds=None):
        #### get prooposal clusters
        if stuff_preds is None:
            object_idxs = torch.nonzero(point_semantic_preds > 1).view(-1)
        else:
            object_idxs = torch.nonzero(stuff_preds == 1).view(-1)
        coords = coords.squeeze()

        batch_idxs_ = batch_idxs[object_idxs]
        batch_offsets_ = utils.get_batch_offsets(batch_idxs_, batch_size)
        coords_ = coords[object_idxs]
        pt_offsets_ = point_offset_preds[object_idxs]

        semantic_preds_cpu = point_semantic_preds[object_idxs].int().cpu()

        if self.cluster_sets == 'Q':
            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_,
                                                                          batch_offsets_, self.cluster_radius,
                                                                          self.cluster_shift_meanActive)
            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu,
                                                                                     idx_shift.cpu(),
                                                                                     start_len_shift.cpu(),
                                                                                     self.cluster_npoint_thre)
            proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
            # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset_shift: (nProposal + 1), int

            proposals_idx = proposals_idx_shift
            proposals_offset = proposals_offset_shift
            scores = torch.ones(proposals_offset_shift.shape[0] - 1, 1).to(point_offset_preds[0].device)

        elif self.cluster_sets == 'P':
            idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.cluster_radius,
                                                              self.cluster_meanActive)
            proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(),
                                                                         self.cluster_npoint_thre)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
            # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int

            proposals_idx = proposals_idx
            proposals_offset = proposals_offset
            scores = torch.ones(proposals_offset.shape[0] - 1, 1).to(point_offset_preds[0].device)

        return scores, proposals_idx, proposals_offset

    def forward(self, input, input_map, coords, rgb, ori_coords, batch_idxs, batch_offsets, epoch):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        raise NotImplementedError()
