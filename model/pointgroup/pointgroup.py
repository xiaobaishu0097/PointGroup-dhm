'''
PointGroup
Written by Li Jiang
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import numpy as np
import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
import sys

sys.path.append('../../')

from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
from model.encoder import pointnet
from model.encoder.unet3d import UNet3D
from model.decoder import decoder
from model.common import coordinate2index, normalize_3d_coordinate


# from model.encoder import PointNetPlusPlus


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
        classes = cfg.classes + 1
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

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        self.model_mode = cfg.model_mode
        self.reso_grid = 32

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        # #### backbone
        # self.input_conv = spconv.SparseSequential(
        #     spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        # )
        #
        # self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block, indice_key_id=1)
        #
        # self.output_layer = spconv.SparseSequential(
        #     norm_fn(m),
        #     nn.ReLU()
        # )

        # #### semantic segmentation
        # self.linear = nn.Linear(m, classes)  # bias(default): True

        # #### offset
        # self.offset = nn.Sequential(
        #     nn.Linear(m, m, bias=True),
        #     norm_fn(m),
        #     nn.ReLU()
        # )
        # self.offset_linear = nn.Linear(m, 3, bias=True)

        # #### score branch
        # self.score_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1)
        # self.score_outputlayer = spconv.SparseSequential(
        #     norm_fn(m),
        #     nn.ReLU()
        # )
        # self.score_linear = nn.Linear(m, 1)
        #
        # self.apply(self.set_bn_init)

        #### pointnet++ encoder
        self.encoder = pointnet.LocalPoolPointnet(
            c_dim=32, dim=6, hidden_dim=32, scatter_type=cfg.scatter_type,
            unet3d=True, unet3d_kwargs={"num_levels": cfg.unet3d_num_levels, "f_maps": 32, "in_channels": 32, "out_channels": 32},
            grid_resolution=32, plane_type='grid',
            padding=0.1, n_blocks=5
        )

        #### center prediction
        self.grid_center_pred = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.grid_center_semantic = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, classes)
        )

        self.grid_center_offset = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        if self.model_mode == 0:
            ### convolutional occupancy networks decoder
            self.decoder = decoder.LocalDecoder(
                dim=3,
                c_dim=32,
                hidden_size=32,
            )

            self.point_offset = nn.Sequential(
                nn.Linear(32, 32, bias=True),
                norm_fn(32),
                nn.ReLU(),
                nn.Linear(32, 3, bias=True),
            )
            self.point_semantic = nn.Linear(32, classes)

        elif self.model_mode == 1:
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(32, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )
            #### semantic segmentation
            self.linear = nn.Linear(m, classes)  # bias(default): True

            #### offset
            self.offset = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
            )
            self.offset_linear = nn.Linear(m, 3, bias=True)
        elif self.model_mode == 2:
            ### convolutional occupancy networks decoder
            self.decoder = decoder.LocalDecoder(
                dim=3,
                c_dim=32,
                hidden_size=32,
            )
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(32, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.unet3d = UNet3D(num_levels=cfg.unet3d_num_levels, f_maps=32, in_channels=32, out_channels=32)

            #### semantic segmentation
            self.linear = nn.Linear(m, classes)  # bias(default): True

            #### offset
            self.offset = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
            )
            self.offset_linear = nn.Linear(m, 3, bias=True)

        #### fix parameter
        # module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
        #               'linear': self.linear, 'offset': self.offset, 'offset_linear': self.offset_linear,
        #               'score_unet': self.score_unet, 'score_outputlayer': self.score_outputlayer,
        #               'score_linear': self.score_linear}
        module_map = {}

        # module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
        #               'linear': self.linear, 'offset': self.offset, 'offset_linear': self.offset_linear,}


        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        #### load pretrain weights
        # self.pretrain_path = '/home/dhm/Auxiliary_Code/PointGroup/exp/scannetv2/pointgroup/pointgroup_run1_scannet/pointgroup_run1_scannet-000000384.pth'
        # self.pretrain_module = ['input_conv', 'unet', 'output_layer', 'linear', 'offset', 'offset_linear']
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
        fea_grid = c.new_zeros(p.size(0), 32, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), 32, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)
        return fea_grid

    def forward(self, input, input_map, coords, rgb, batch_idxs, batch_offsets, epoch):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        encoded_feats = self.encoder(coords.unsqueeze(dim=0), rgb.unsqueeze(dim=0))
        '''encoded_feats:{
        coord: coordinate of points (b, n, 3)
        index: index of points (b, 1, n)
        point: feature of points (b, n, 32)
        grid: grid features (b, c, grid_dim, grid_dim, grid_dim)
        }
        '''

        if self.model_mode == 0:
            point_feats = self.decoder(encoded_feats['coord'], encoded_feats).squeeze(dim=0)

            point_offset_preds = self.point_offset(point_feats)
            point_semantic_preds = self.point_semantic(point_feats)
        elif self.model_mode == 1:
            voxel_feats = pointgroup_ops.voxelization(
                encoded_feats['point'].squeeze(dim=0),
                # input['pt_feats'],
                input['v2p_map'],
                input['mode']
            )  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(
                voxel_feats, input['voxel_coords'], input['spatial_shape'], input['batch_size']
            )
            output = self.input_conv(input_)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]

            #### semantic segmentation
            semantic_scores = self.linear(output_feats)  # (N, nClass), float
            point_semantic_preds = semantic_scores

            # ret['semantic_scores'] = semantic_scores

            #### offset
            pt_offsets = self.offset(output_feats) # (N, 3), float32
            pt_offsets = self.offset_linear(pt_offsets)
            point_offset_preds = pt_offsets

            # ret['pt_offsets'] = pt_offsets
        elif self.model_mode == 2:
            voxel_feats = pointgroup_ops.voxelization(
                encoded_feats['point'].squeeze(dim=0),
                # input['pt_feats'],
                input['v2p_map'],
                input['mode']
            )  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(
                voxel_feats, input['voxel_coords'], input['spatial_shape'], input['batch_size']
            )
            output = self.input_conv(input_)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]

            grid_feats = self.generate_grid_features(encoded_feats['coord'], encoded_feats['point'] + output_feats.unsqueeze(dim=0))
            grid_feats = self.unet3d(grid_feats)
            encoded_feats['grid'] = grid_feats

            #### semantic segmentation
            semantic_scores = self.linear(output_feats)  # (N, nClass), float
            point_semantic_preds = semantic_scores

            #### offset
            pt_offsets = self.offset(output_feats)  # (N, 3), float32
            pt_offsets = self.offset_linear(pt_offsets)
            point_offset_preds = pt_offsets

        bs, c_dim, grid_size = encoded_feats['grid'].shape[0], encoded_feats['grid'].shape[1], encoded_feats['grid'].shape[2]
        grid_feats = encoded_feats['grid'].reshape(bs, c_dim, -1).permute(0, 2, 1)

        ### grid point center probabilty prediction
        grid_center_preds = self.grid_center_pred(grid_feats)

        grid_center_semantic_preds = self.grid_center_semantic(grid_feats)

        ### grid point center offset vector prediction
        grid_center_offset_preds = self.grid_center_offset(grid_feats)

        ret['point_semantic_preds'] = point_semantic_preds
        ret['point_offset_preds'] = point_offset_preds
        ret['grid_center_preds'] = grid_center_preds
        ret['grid_center_semantic_preds'] = grid_center_semantic_preds
        ret['grid_center_offset_preds'] = grid_center_offset_preds

        return ret


def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()
    center_criterion = WeightedFocalLoss(alpha=cfg.focal_loss_alpha, gamma=cfg.focal_loss_gamma).cuda()
    center_semantic_criterion = nn.CrossEntropyLoss().cuda()
    center_offset_criterion = nn.L1Loss().cuda()

    def test_model_fn(batch, model, epoch):
        coords = batch['locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()  # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()  # (N, C), float32, cuda

        instance_info = batch['instance_info'].cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        labels = batch['labels'].cuda()  # (N), long, cuda
        instance_centers = batch['instance_centers'].cuda()
        grid_center_gt = batch['grid_center_gt'].cuda()
        grid_center_offset = batch['grid_center_offset'].cuda()
        grid_instance_label = batch['grid_instance_label'].cuda()
        grid_xyz = batch['grid_xyz'].cuda()

        batch_offsets = batch['offsets'].cuda()  # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        rgb = feats

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
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

        ret = model(input_, p2v_map, coords_float, rgb, coords[:, 0].int(), batch_offsets, epoch)

        point_offset_preds = ret['point_offset_preds']
        point_offset_preds = point_offset_preds.squeeze()

        point_semantic_preds = ret['point_semantic_preds']
        point_semantic_preds = point_semantic_preds.squeeze()

        grid_center_preds = ret['grid_center_preds']
        grid_center_preds = grid_center_preds.reshape(1, -1)

        grid_center_semantic_preds = ret['grid_center_semantic_preds']
        grid_center_semantic_preds = grid_center_semantic_preds.squeeze(dim=0)
        grid_center_semantic_preds = grid_center_semantic_preds.max(dim=1)[1]

        grid_center_offset_preds = ret['grid_center_offset_preds'] # (1, 32**3, 3)
        grid_center_offset_preds = grid_center_offset_preds.squeeze(dim=0)

        ### only be used during debugging
        # point_offset_preds = instance_info[:, 0:3] - coords_float

        # point_semantic_preds = labels
        #
        # fake_grid_center = torch.zeros_like(grid_center_preds)
        # fake_grid_center[0, grid_center_gt.long()] = 1
        # grid_center_preds = fake_grid_center
        #
        # grid_center_offset_preds[grid_center_gt.long(), :] = grid_center_offset
        #
        # grid_center_semantic_preds = grid_instance_label

        ##### preds
        with torch.no_grad():
            preds = {}
            preds['pt_offsets'] = point_offset_preds
            preds['semantic'] = point_semantic_preds
            preds['grid_center_preds'] = grid_center_preds
            preds['grid_center_semantic_preds'] = grid_center_semantic_preds
            preds['grid_center_offset_preds'] = grid_center_offset_preds
            preds['grid_center_gt'] = grid_center_gt
            preds['grid_center_offset'] = grid_center_offset
            preds['pt_coords'] = coords_float
            preds['grid_xyz'] = grid_xyz
            preds['instance_centers'] = instance_centers
            # preds['semantic'] = semantic_scores
            # preds['pt_offsets'] = pt_offsets
            # if (epoch > cfg.prepare_epochs):
            #     preds['score'] = scores
            #     preds['proposals'] = (proposals_idx, proposals_offset)

        return preds

    def model_fn(batch, model, epoch):
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()  # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()  # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()  # (N, C), float32, cuda
        labels = batch['labels'].cuda()  # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda()  # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch['instance_info'].cuda()  # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda
        instance_centers = batch['instance_centers'].cuda()

        instance_heatmap = batch['instance_heatmap'].cuda()
        grid_center_gt = batch['grid_center_gt'].cuda()
        grid_center_offset = batch['grid_center_offset'].cuda()
        grid_instance_label = batch['grid_instance_label'].cuda()

        batch_offsets = batch['offsets'].cuda()  # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        rgb = feats

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
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

        ret = model(input_, p2v_map, coords_float, rgb, coords[:, 0].int(), batch_offsets, epoch)

        point_offset_preds = ret['point_offset_preds']
        point_offset_preds = point_offset_preds.squeeze()

        point_semantic_preds = ret['point_semantic_preds']
        point_semantic_preds = point_semantic_preds.squeeze()

        grid_center_preds = ret['grid_center_preds'] # (1, 32**3)
        grid_center_preds = grid_center_preds.reshape(1, -1)

        grid_center_semantic_preds = ret['grid_center_semantic_preds']
        grid_center_semantic_preds = grid_center_semantic_preds.squeeze(dim=0)

        grid_center_offset_preds = ret['grid_center_offset_preds'] # (1, 32**3, 3)
        grid_center_offset_preds = grid_center_offset_preds.squeeze(dim=0)
        grid_center_offset_preds = grid_center_offset_preds[grid_center_gt.long(), :] # (nInst, 3)

        # instance_heatmap = torch.zeros((32**3)).cuda()
        # instance_heatmap[grid_center_gt.long()] = 1

        instance_heatmap = instance_heatmap.reshape((1, -1))

        # grid_coords = normalize_3d_coordinate(
        #     torch.cat((coords_float, instance_centers), dim=0).unsqueeze(dim=0).clone(), padding=0.1)
        # center_indexs = coordinate2index(grid_coords, 32, coord_type='3d')[:, :, -instance_centers.shape[0]:]
        # grid_gt_centers = torch.zeros_like(grid_centers).cuda()
        # grid_gt_centers[:, center_indexs] = 1

        # semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda
        # pt_offsets = ret['pt_offsets']  # (N, 3), float32, cuda
        # if (epoch > cfg.prepare_epochs):
        #     scores, proposals_idx, proposals_offset = ret['proposal_scores']
            # scores: (nProposal, 1) float, cuda
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu

        loss_inp = {}

        loss_inp['pt_offsets'] = (point_offset_preds, coords_float, instance_info, instance_labels)
        loss_inp['semantic_scores'] = (point_semantic_preds, labels)

        loss_inp['grid_centers'] = (grid_center_preds, instance_heatmap)
        loss_inp['grid_center_semantics'] = (grid_center_semantic_preds, grid_instance_label)
        loss_inp['grid_center_offsets'] = (grid_center_offset_preds, grid_center_offset)
        # loss_inp['semantic_scores'] = (semantic_scores, labels)
        # loss_inp['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)
        # if (epoch > cfg.prepare_epochs):
        #     loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)

        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['grid_center_preds'] = grid_center_preds
            preds['grid_center_offset_preds'] = grid_center_offset_preds
            preds['grid_center_gt'] = grid_center_gt
            preds['instance_heatmap'] = instance_heatmap
            preds['grid_center_offset'] = grid_center_offset
            # preds['semantic'] = semantic_scores
            # preds['pt_offsets'] = pt_offsets
            # if (epoch > cfg.prepare_epochs):
            #     preds['score'] = scores
            #     preds['proposals'] = (proposals_idx, proposals_offset)

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

        ### center loss
        grid_center_preds, instance_heatmap = loss_inp['grid_centers']

        center_loss = center_criterion(grid_center_preds, instance_heatmap)
        loss_out['center_loss'] = (center_loss, instance_heatmap.shape[-1])

        ### center semantic loss
        grid_center_semantic_preds, grid_instance_label = loss_inp['grid_center_semantics']
        grid_valid_index = instance_heatmap.squeeze(dim=0) > 0
        center_semantic_loss = center_semantic_criterion(
            grid_center_semantic_preds[grid_valid_index, :], grid_instance_label[grid_valid_index].to(torch.long)
        )

        ### center offset loss
        grid_center_offset_preds, grid_center_offsets = loss_inp['grid_center_offsets']

        center_offset_loss = center_offset_criterion(grid_center_offset_preds, grid_center_offsets)
        loss_out['center_offset_loss'] = (center_offset_loss, grid_center_offsets.shape[0])

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords  # (N, 3)
        pt_diff = pt_offsets - gt_offsets  # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
        valid = (instance_labels != cfg.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        # if (epoch > cfg.prepare_epochs):
        #     '''score loss'''
        #     scores, proposals_idx, proposals_offset, instance_pointnum = loss_inp['proposal_scores']
        #     # scores: (nProposal, 1), float32
        #     # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        #     # proposals_offset: (nProposal + 1), int, cpu
        #     # instance_pointnum: (total_nInst), int
        #
        #     ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels,
        #                                   instance_pointnum)  # (nProposal, nInstance), float
        #     gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
        #     gt_scores = get_segmented_scores(gt_ious, cfg.fg_thresh, cfg.bg_thresh)
        #
        #     score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
        #     score_loss = score_loss.mean()
        #
        #     loss_out['score_loss'] = (score_loss, gt_ious.shape[0])

        '''total loss'''
        # loss = cfg.loss_weight[0] * semantic_loss + cfg.loss_weight[1] * offset_norm_loss + cfg.loss_weight[
        #     2] * offset_dir_loss
        # if (epoch > cfg.prepare_epochs):
        #     loss += (cfg.loss_weight[3] * score_loss)
        loss = cfg.loss_weight[0] * center_loss + cfg.loss_weight[1] * center_semantic_loss + \
               cfg.loss_weight[2] * center_offset_loss + cfg.loss_weight[2] * semantic_loss + \
               cfg.loss_weight[3] * offset_norm_loss + cfg.loss_weight[4] * offset_dir_loss

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
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()