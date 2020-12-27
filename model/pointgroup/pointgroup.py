'''
PointGroup
Written by Li Jiang
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import os
import numpy as np
import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
import sys

sys.path.append('../../')

from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
from model.encoder import pointnet, pointnetpp
from model.encoder.unet3d import UNet3D
from model.decoder import decoder
from model.common import coordinate2index, normalize_3d_coordinate

from model.encoder.unet3d import UNet3D
from model.Pointnet2.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModule


class backbone_pointnet2(nn.Module):
    def __init__(self):
        super(backbone_pointnet2, self).__init__()
        self.sa1 = PointnetSAModule(mlp=[6, 32, 32, 64], npoint=1024, radius=0.1, nsample=32, bn=True)
        self.sa2 = PointnetSAModule(mlp=[64, 64, 64, 128], npoint=256, radius=0.2, nsample=64, bn=True)
        self.sa3 = PointnetSAModule(mlp=[128, 128, 128, 256], npoint=64, radius=0.4, nsample=128, bn=True)
        self.sa4 = PointnetSAModule(mlp=[256, 256, 256, 512], npoint=None, radius=None, nsample=None, bn=True)
        self.fp4 = PointnetFPModule(mlp=[768, 256, 256])
        self.fp3 = PointnetFPModule(mlp=[384, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[320, 256, 128])
        # self.fp1 = PointnetFPModule(mlp=[137, 128, 128, 128, 128])
        self.fp1 = PointnetFPModule(mlp=[137, 128, 128, 64, 32])

    def forward(self, xyz, points):
        l1_xyz, l1_points = self.sa1(xyz.contiguous(), points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz.contiguous(), l1_xyz, torch.cat((xyz.transpose(1, 2), points), dim=1), l1_points)

        # global_features = l4_points.view(-1, 512)
        global_features = l3_points.view(-1, 512)
        point_features = l0_points.transpose(1, 2)

        return point_features, global_features


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
                  for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False,
                                    indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id + 1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False,
                                           indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn,
                                                         indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output


class PointGroup(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_channel
        m = cfg.m
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

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

        self.backbone = cfg.backbone
        self.model_mode = cfg.model_mode
        self.reso_grid = 32
        self.cluster_sets = cfg.cluster_sets
        self.m = cfg.m

        self.pointnet_include_rgb = cfg.pointnet_include_rgb

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        self.unet3d = None

        ### Our target model, based on Panoptic Deeplab
        if self.model_mode == 0 or self.model_mode == 'Zheng_panoptic_wpointnet_PointGroup':
            if self.backbone == 'pointnet':
                #### PointNet backbone encoder
                self.pointnet_encoder = pointnet.LocalPoolPointnet(
                    c_dim=m, dim=6, hidden_dim=m, scatter_type=cfg.scatter_type, grid_resolution=32,
                    plane_type='grid', padding=0.1, n_blocks=5
                )
            elif self.backbone == 'pointnet++_yanx':
                self.pointnet_encoder = pointnetpp.PointNetPlusPlus(
                    c_dim=self.m, include_rgb=self.pointnet_include_rgb
                )

            elif self.backbone == 'pointnet++_shi':
                self.pointnet_encoder = backbone_pointnet2()

            self.unet3d = UNet3D(
                num_levels=cfg.unet3d_num_levels, f_maps=m, in_channels=m, out_channels=m
            )

            ### point prediction branch
            ### sparse 3D U-Net
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(m, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            ### centre prediction branch
            ### convolutional occupancy networks decoder
            self.centre_decoder = decoder.LocalDecoder(
                dim=3,
                c_dim=32,
                hidden_size=32,
            )

        ### only the upper branch of our target model
        elif self.model_mode == 'Zheng_upper_wpointnet_PointGroup':
            #### PointNet backbone encoder
            if self.backbone == 'pointnet':
                #### PointNet backbone encoder
                self.pointnet_encoder = pointnet.LocalPoolPointnet(
                    c_dim=m, dim=6, hidden_dim=m, scatter_type=cfg.scatter_type, grid_resolution=32,
                    plane_type='grid', padding=0.1, n_blocks=5
                )
            elif self.backbone == 'pointnet++_yanx':
                self.pointnet_encoder = pointnetpp.PointNetPlusPlus(
                    c_dim=self.m, include_rgb=self.pointnet_include_rgb
                )
            elif self.backbone == 'pointnet++_shi':
                self.pointnet_encoder = backbone_pointnet2()

            ### point prediction branch
            ### sparse 3D U-Net
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(m, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

        ### Our target model without PointNet encoder
        ### Ablation study: figure out the performance improvement of PointNet encoder
        elif self.model_mode == 'Zheng_panoptic_wopointnet_PointGroup':
            ### point prediction branch
            ### sparse 3D U-Net
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            ### centre prediction branch
            #### PointNet encoder
            if self.backbone == 'pointnet':
                #### PointNet backbone encoder
                self.pointnet_encoder = pointnet.LocalPoolPointnet(
                    c_dim=m, dim=6, hidden_dim=m, scatter_type=cfg.scatter_type, grid_resolution=32,
                    plane_type='grid', padding=0.1, n_blocks=5
                )
            elif self.backbone == 'pointnet++_yanx':
                self.pointnet_encoder = pointnetpp.PointNetPlusPlus(
                    c_dim=self.m, include_rgb=self.pointnet_include_rgb
                )

            elif self.backbone == 'pointnet++_shi':
                self.pointnet_encoder = backbone_pointnet2()

            self.unet3d = UNet3D(
                num_levels=cfg.unet3d_num_levels, f_maps=m, in_channels=m, out_channels=m
            )

            ### convolutional occupancy networks decoder
            self.centre_decoder = decoder.LocalDecoder(
                dim=3,
                c_dim=32,
                hidden_size=32,
            )

        ### only the upper branch of our target model without PointNet encoder
        elif self.model_mode == 'Zheng_upper_wopointnet_PointGroup':
            ### point prediction branch
            ### sparse 3D U-Net
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

        ### same network architecture as PointGroup
        elif self.model_mode == 'Jiang_original_PointGroup':
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            #### score branch
            self.score_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1)
            self.score_outputlayer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )
            self.score_linear = nn.Linear(m, 1)

            self.apply(self.set_bn_init)

        ### point prediction
        self.point_offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU(),
            nn.Linear(m, 3, bias=True),
        )

        self.point_semantic = nn.Linear(m, classes)

        #### centre prediction
        ## centre probability
        self.centre_pred = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(),
            nn.Linear(m, 1)
        )

        ## centre semantic
        self.centre_semantic = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(),
            nn.Linear(m, classes)
        )

        ## centre offset
        self.centre_offset = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(),
            nn.Linear(m, 3)
        )

        #### fix parameter
        module_map = {}

        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        if self.pretrain_path is not None:
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print(
                    "Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m))

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
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()],
                                    1)  # (sumNPoint, 1 + 3)

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

        if self.backbone == 'pointnet':
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

        elif self.backbone == 'pointnet++_yanx':
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

        elif self.backbone == 'pointnet++_shi':
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

    def pointgroup_cluster_algorithm(self, coords, point_offset_preds, point_semantic_preds, batch_idxs, batch_size):
        #### get prooposal clusters
        object_idxs = torch.nonzero(point_semantic_preds > 1).view(-1)
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
            scores = torch.ones(proposals_offset_shift.shape[0] - 1, 1).to(point_offset_preds.device)

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
            scores = torch.ones(proposals_offset.shape[0] - 1, 1).to(point_offset_preds.device)

        return scores, proposals_idx, proposals_offset

    def forward(self, input, input_map, coords, rgb, ori_coords, batch_idxs, batch_offsets, epoch):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        batch_idxs = batch_idxs.squeeze()

        if self.model_mode == 0:
            point_feats, grid_feats = self.pointnet_backbone_forward(coords, ori_coords, rgb, batch_offsets)

            voxel_feats = pointgroup_ops.voxelization(point_feats, input['v2p_map'], input['mode'])  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(
                voxel_feats, input['voxel_coords'],
                input['spatial_shape'], input['batch_size']
            )
            output = self.input_conv(input_)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]
            output_feats = output_feats.squeeze(dim=0)

            ### point prediction
            #### point semantic label prediction
            semantic_scores = self.point_semantic(output_feats)  # (N, nClass), float
            # point_semantic_preds = semantic_scores
            point_semantic_preds = semantic_scores.max(1)[1]
            #### point offset prediction
            point_offset_preds = self.point_offset(output_feats)  # (N, 3), float32

            ### centre prediction
            bs, c_dim, grid_size = grid_feats.shape[0], grid_feats.shape[1], grid_feats.shape[2]
            grid_feats = grid_feats.reshape(bs, c_dim, -1).permute(0, 2, 1)

            centre_preds = self.centre_pred(grid_feats)
            centre_semantic_preds = self.centre_semantic(grid_feats)
            centre_offset_preds = self.centre_offset(grid_feats)

            ret['point_semantic_scores'] = semantic_scores
            ret['point_offset_preds'] = point_offset_preds

            ret['centre_preds'] = centre_preds
            ret['centre_semantic_preds'] = centre_semantic_preds
            ret['centre_offset_preds'] = centre_offset_preds

        elif self.model_mode == 'Zheng_panoptic_wpointnet_PointGroup':
            point_feats, grid_feats = self.pointnet_backbone_forward(coords, ori_coords, rgb, batch_offsets)

            voxel_feats = pointgroup_ops.voxelization(point_feats, input['v2p_map'], input['mode'])  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(
                voxel_feats, input['voxel_coords'],
                input['spatial_shape'], input['batch_size']
            )
            output = self.input_conv(input_)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]
            output_feats = output_feats.squeeze(dim=0)

            ### point prediction
            #### point semantic label prediction
            semantic_scores = self.point_semantic(output_feats)  # (N, nClass), float
            # point_semantic_preds = semantic_scores
            point_semantic_preds = semantic_scores.max(1)[1]
            #### point offset prediction
            point_offset_preds = self.point_offset(output_feats)  # (N, 3), float32

            ### centre prediction
            bs, c_dim, grid_size = grid_feats.shape[0], grid_feats.shape[1], grid_feats.shape[2]
            grid_feats = grid_feats.reshape(bs, c_dim, -1).permute(0, 2, 1)

            centre_preds = self.centre_pred(grid_feats)
            centre_semantic_preds = self.centre_semantic(grid_feats)
            centre_offset_preds = self.centre_offset(grid_feats)

            if (epoch == self.test_epoch):
                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_preds, point_semantic_preds, batch_idxs, input['batch_size']
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = semantic_scores
            ret['point_offset_preds'] = point_offset_preds

            ret['centre_preds'] = centre_preds
            ret['centre_semantic_preds'] = centre_semantic_preds
            ret['centre_offset_preds'] = centre_offset_preds

        elif self.model_mode == 'Zheng_upper_wpointnet_PointGroup':
            point_feats, _ = self.pointnet_backbone_forward(coords, ori_coords, rgb, batch_offsets)

            voxel_feats = pointgroup_ops.voxelization(point_feats, input['v2p_map'], input['mode'])  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(
                voxel_feats, input['voxel_coords'],
                input['spatial_shape'], input['batch_size']
            )
            output = self.input_conv(input_)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]
            output_feats = output_feats.squeeze(dim=0)

            ### point prediction
            #### point semantic label prediction
            semantic_scores = self.point_semantic(output_feats)  # (N, nClass), float
            # point_semantic_preds = semantic_scores
            point_semantic_preds = semantic_scores.max(1)[1]

            #### point offset prediction
            point_offset_preds = self.point_offset(output_feats)  # (N, 3), float32

            if (epoch == self.test_epoch):
                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_preds, point_semantic_preds, batch_idxs, input['batch_size']
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = semantic_scores
            ret['point_offset_preds'] = point_offset_preds

        elif self.model_mode == 'Zheng_panoptic_wopointnet_PointGroup':
            voxel_feats = pointgroup_ops.voxelization(input['pt_feats'], input['v2p_map'], input['mode'])  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(
                voxel_feats, input['voxel_coords'],
                input['spatial_shape'], input['batch_size']
            )
            output = self.input_conv(input_)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]
            output_feats = output_feats.squeeze(dim=0)

            ### point prediction
            #### point semantic label prediction
            semantic_scores = self.point_semantic(output_feats)  # (N, nClass), float
            # point_semantic_preds = semantic_scores
            point_semantic_preds = semantic_scores.max(1)[1]

            #### point offset prediction
            point_offset_preds = self.point_offset(output_feats)  # (N, 3), float32

            ### centre prediction
            # encoded_feats = self.pointnet_centre_encoder(coords.unsqueeze(dim=0), rgb.unsqueeze(dim=0))
            # bs, c_dim, grid_size = encoded_feats['grid'].shape[0], encoded_feats['grid'].shape[1], \
            #                        encoded_feats['grid'].shape[2]
            # grid_feats = encoded_feats['grid'].reshape(bs, c_dim, -1).permute(0, 2, 1)
            _, grid_feats = self.pointnet_backbone_forward(coords, ori_coords, rgb, batch_offsets)
            bs, c_dim, grid_size = grid_feats.shape[0], grid_feats.shape[1], grid_feats.shape[2]
            grid_feats = grid_feats.reshape(bs, c_dim, -1).permute(0, 2, 1)

            centre_preds = self.centre_pred(grid_feats)
            centre_semantic_preds = self.centre_semantic(grid_feats)
            centre_offset_preds = self.centre_offset(grid_feats)

            if (epoch == self.test_epoch):
                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_preds, point_semantic_preds, batch_idxs, input['batch_size']
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = semantic_scores
            ret['point_offset_preds'] = point_offset_preds

            ret['centre_preds'] = centre_preds
            ret['centre_semantic_preds'] = centre_semantic_preds
            ret['centre_offset_preds'] = centre_offset_preds

        elif self.model_mode == 'Zheng_upper_wopointnet_PointGroup':
            voxel_feats = pointgroup_ops.voxelization(input['pt_feats'], input['v2p_map'], input['mode'])  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(
                voxel_feats, input['voxel_coords'],
                input['spatial_shape'], input['batch_size']
            )
            output = self.input_conv(input_)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]
            output_feats = output_feats.squeeze(dim=0)

            ### point prediction
            #### point semantic label prediction
            semantic_scores = self.point_semantic(output_feats)  # (N, nClass), float
            # point_semantic_preds = semantic_scores
            point_semantic_preds = semantic_scores.max(1)[1]

            #### point offset prediction
            point_offset_preds = self.point_offset(output_feats)  # (N, 3), float32

            if (epoch == self.test_epoch):
                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_preds, point_semantic_preds, batch_idxs, input['batch_size']
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = semantic_scores
            ret['point_offset_preds'] = point_offset_preds

        elif self.model_mode == 'Jiang_original_PointGroup':
            voxel_feats = pointgroup_ops.voxelization(input['pt_feats'], input['v2p_map'], input['mode'])  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(
                voxel_feats, input['voxel_coords'], input['spatial_shape'], input['batch_size']
            )
            output = self.input_conv(input_)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]
            output_feats = output_feats.squeeze(dim=0)

            ### point prediction
            #### point semantic label prediction
            semantic_scores = self.point_semantic(output_feats)  # (N, nClass), float
            point_semantic_preds = semantic_scores.max(1)[1]

            #### point offset prediction
            point_offset_preds = self.point_offset(output_feats)  # (N, 3), float32

            if (epoch > self.prepare_epochs):
                #### get prooposal clusters
                object_idxs = torch.nonzero(point_semantic_preds > 1).view(-1)

                batch_idxs_ = batch_idxs[object_idxs]
                batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input['batch_size'])
                coords_ = coords[object_idxs]
                pt_offsets_ = point_offset_preds[object_idxs]

                semantic_preds_cpu = point_semantic_preds[object_idxs].int().cpu()

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

                idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.cluster_radius,
                                                                  self.cluster_meanActive)
                proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(),
                                                                             self.cluster_npoint_thre)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int

                proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
                proposals_offset_shift += proposals_offset[-1]
                proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
                proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))

                #### proposals voxelization again
                input_feats, inp_map = self.clusters_voxelization(proposals_idx, proposals_offset, output_feats, coords,
                                                                  self.score_fullscale, self.score_scale, self.mode)

                #### score
                score = self.score_unet(input_feats)
                score = self.score_outputlayer(score)
                score_feats = score.features[inp_map.long()]  # (sumNPoint, C)
                score_feats = pointgroup_ops.roipool(score_feats, proposals_offset.cuda())  # (nProposal, C)
                scores = self.score_linear(score_feats)  # (nProposal, 1)

                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = semantic_scores
            ret['point_offset_preds'] = point_offset_preds

        return ret


def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
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
    score_criterion = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch):
        coords = batch['locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()  # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        ori_coords = batch['ori_coords'].cuda()
        feats = batch['feats'].cuda()  # (N, C), float32, cuda

        batch_offsets = batch['offsets'].cuda()  # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        rgb = feats

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), -1)
        # voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda
        #
        # input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)
        input_ = {
            'pt_feats': feats,
            'v2p_map': v2p_map,
            'mode': cfg.mode,
            'voxel_coords': voxel_coords.int(),
            'spatial_shape': spatial_shape,
            'batch_size': cfg.batch_size,
        }

        ret = model(input_, p2v_map, coords_float, rgb, ori_coords, coords[:, :, 0].int(), batch_offsets, epoch)

        point_offset_preds = ret['point_offset_preds']
        point_offset_preds = point_offset_preds.squeeze()

        point_semantic_scores = ret['point_semantic_scores']
        point_semantic_scores = point_semantic_scores.squeeze()

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
        # instance_info = batch['instance_info'].squeeze(dim=0).cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        # labels = batch['labels'].squeeze(dim=0).cuda()  # (N), long, cuda
        # grid_centre_gt = batch['grid_centre_gt'].squeeze(dim=0).cuda()
        # grid_centre_offset = batch['grid_centre_offset'].squeeze(dim=0).cuda()
        # grid_instance_label = batch['grid_instance_label'].squeeze(dim=0).cuda()

        # point_offset_preds = instance_info[:, 0:3] - coords_float
        #
        # point_semantic_preds = labels
        #
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
        instance_centres = batch['instance_centres'].cuda()

        instance_heatmap = batch['grid_centre_heatmap'].cuda()
        grid_centre_gt = batch['grid_centre_indicator'].cuda()
        centre_offset_labels = batch['grid_centre_offset_labels'].cuda()
        centre_semantic_labels = batch['grid_centre_semantic_labels'].cuda()

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

        input_ = {
            'pt_feats': feats,
            'v2p_map': v2p_map,
            'mode': cfg.mode,
            'voxel_coords': voxel_coords.int(),
            'spatial_shape': spatial_shape,
            'batch_size': cfg.batch_size,
        }

        ret = model(input_, p2v_map, coords_float, rgb, ori_coords, coords[:, 0].int(), batch_offsets, epoch)

        point_offset_preds = ret['point_offset_preds']
        point_offset_preds = point_offset_preds.squeeze()

        point_semantic_scores = ret['point_semantic_scores']
        point_semantic_scores = point_semantic_scores.squeeze()

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

        loss_inp['pt_offsets'] = (
            point_offset_preds, coords_float.squeeze(dim=0), instance_info.squeeze(dim=0),
            instance_centres.squeeze(dim=0), instance_labels.squeeze(dim=0)
        )
        loss_inp['semantic_scores'] = (point_semantic_scores, labels.squeeze(dim=0))

        if 'centre_preds' in ret.keys():
            loss_inp['centre_preds'] = (centre_preds, instance_heatmap)
            loss_inp['centre_semantic_preds'] = (centre_semantic_preds, centre_semantic_labels)
            loss_inp['centre_offset_preds'] = (centre_offset_preds, centre_offset_labels[:, 1:])

        if (epoch > cfg.prepare_epochs) and (cfg.model_mode == 'Jiang_original_PointGroup'):
            loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)

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
            if (epoch > cfg.prepare_epochs) and (cfg.model_mode == 'Jiang_original_PointGroup'):
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

        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_centre, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords  # (N, 3)
        pt_valid_index = instance_labels != cfg.ignore_label
        if cfg.offset_norm_criterion == 'l1':
            pt_diff = pt_offsets - gt_offsets  # (N, 3)
            pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
            valid = (instance_labels != cfg.ignore_label).float()
            offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)
        if cfg.offset_norm_criterion == 'l2':
            offset_norm_loss = offset_norm_criterion(pt_offsets[pt_valid_index], gt_offsets[pt_valid_index])
        elif cfg.offset_norm_criterion == 'triplet':
            ### offset l1 distance loss: learn to move towards the true instance centre
            offset_norm_loss = offset_norm_criterion(pt_offsets[pt_valid_index], gt_offsets[pt_valid_index])

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
            shifted_coords = coords + pt_offsets
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
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        if 'centre_preds' in loss_inp.keys():
            '''centre loss'''
            centre_preds, instance_heatmap = loss_inp['centre_preds']

            centre_loss = centre_criterion(centre_preds, instance_heatmap)
            loss_out['centre_loss'] = (centre_loss, instance_heatmap.shape[-1])

            '''centre semantic loss'''
            centre_semantic_preds, grid_instance_label = loss_inp['centre_semantic_preds']
            grid_valid_index = instance_heatmap > 0
            centre_semantic_loss = []
            for sample_index in range(instance_heatmap.shape[0]):
                if sum(grid_valid_index[sample_index, :]) != 0:
                    centre_semantic_loss.append(
                        centre_semantic_criterion(
                            centre_semantic_preds[sample_index, grid_valid_index[sample_index, :], :],
                            grid_instance_label[sample_index, grid_valid_index[sample_index, :]].to(torch.long)
                        )
                    )
            if len(centre_semantic_loss) != 0:
                centre_semantic_loss = torch.mean(torch.stack(centre_semantic_loss))
            else:
                centre_semantic_loss = 0

            loss_out['centre_semantic_loss'] = (centre_semantic_loss, grid_instance_label.shape[0])

            '''centre offset loss'''
            centre_offset_preds, grid_centre_offsets = loss_inp['centre_offset_preds']

            centre_offset_loss = centre_offset_criterion(centre_offset_preds, grid_centre_offsets)
            loss_out['centre_offset_loss'] = (centre_offset_loss, grid_centre_offsets.shape[0])

        if (epoch > cfg.prepare_epochs) and (cfg.model_mode == 'Jiang_original_PointGroup'):
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

        '''total loss'''
        loss = cfg.loss_weight[3] * semantic_loss + cfg.loss_weight[4] * offset_norm_loss + \
               cfg.loss_weight[5] * offset_dir_loss
        if cfg.offset_norm_criterion == 'triplet':
            loss += cfg.loss_weight[6] * offset_norm_triplet_loss
        if 'centre_preds' in loss_inp.keys():
            loss += cfg.loss_weight[0] * centre_loss + cfg.loss_weight[1] * centre_semantic_loss + \
                    cfg.loss_weight[2] * centre_offset_loss
        if (epoch > cfg.prepare_epochs) and (cfg.model_mode == 'Jiang_original_PointGroup'):
            loss += (1 * score_loss)

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


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=0.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        at = self.alpha.gather(0, targets.data.view(-1).to(torch.long))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt.view(-1))**self.gamma * BCE_loss.view(-1)
        return F_loss.mean()