import time

import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import numpy as np
import spconv
import functools
import sys, os

sys.path.append('../')

from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils

from model.encoder import pointnet, pointnetpp
from model.encoder.unet3d import UNet3D
from model.decoder import decoder
from model.common import coordinate2index, normalize_3d_coordinate

from model.components import ResidualBlock, VGGBlock, UBlock
from model.components import backbone_pointnet2, backbone_pointnet2_deeper

from model.Pointnet2.pointnet2 import pointnet2_utils
from model.components import ProposalTransformer, euclidean_dist


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
        self.proposal_refinement = cfg.proposal_refinement
        self.point_xyz_reconstruction_loss = cfg.point_xyz_reconstruction_loss
        self.point_rgb_reconstruction_loss = cfg.point_rgb_reconstruction_loss

        self.pointnet_max_npoint = 8196

        self.full_scale = cfg.full_scale
        self.batch_size = cfg.batch_size
        self.instance_triplet_loss = cfg.instance_triplet_loss

        self.instance_classifier = cfg.instance_classifier
        self.voxel_center_prediction = cfg.voxel_center_prediction

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords == True:
            input_c += 3
        elif cfg.use_coords == 'Z':
            input_c += 1

        self.unet3d = None

        ### Our target model, based on Panoptic Deeplab
        if self.model_mode == 'Center_clustering':
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
                self.pointnet_encoder = backbone_pointnet2(output_dim=m)

            self.unet3d = UNet3D(
                num_levels=cfg.unet3d_num_levels, f_maps=m, in_channels=m, out_channels=m
            )

            ### point prediction branch
            ### sparse 3D U-Net
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(m, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, backbone=True, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            ### center prediction branch
            ### convolutional occupancy networks decoder
            self.center_decoder = decoder.LocalDecoder(
                dim=3,
                c_dim=32,
                hidden_size=32,
            )

            self.point_offset = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, 3, bias=True),
            )

            self.point_semantic = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, classes, bias=True),
            )

            self.center_pred = nn.Sequential(
                nn.Linear(m, m),
                nn.ReLU(),
                nn.Linear(m, 1)
            )

            self.center_semantic = nn.Sequential(
                nn.Linear(m, m),
                nn.ReLU(),
                nn.Linear(m, classes)
            )

            self.center_offset = nn.Sequential(
                nn.Linear(m, m),
                nn.ReLU(),
                nn.Linear(m, 3)
            )

            module_map = {
                'module.point_offset': self.point_offset,
            }

        elif self.model_mode == 'Center_pointnet++_clustering':
            self.center_clustering = cfg.center_clustering

            self.pointnet_encoder = backbone_pointnet2(output_dim=m)

            ### point prediction branch
            ### sparse 3D U-Net
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, backbone=True, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.point_offset = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, 3, bias=True),
            )

            self.point_semantic = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, classes, bias=True),
            )

            self.center_pred = nn.Sequential(
                nn.Linear(m, m),
                nn.ReLU(),
                nn.Linear(m, 1)
            )

            self.center_semantic = nn.Sequential(
                nn.Linear(m, m),
                nn.ReLU(),
                nn.Linear(m, classes)
            )

            self.center_offset = nn.Sequential(
                nn.Linear(m, m),
                nn.ReLU(),
                nn.Linear(m, 3)
            )

            module_map = {
                'input_conv': self.input_conv,
                'unet': self.unet,
                'output_layer': self.output_layer,
                'point_offset': self.point_offset,
                'point_semantic': self.point_semantic,
                'pointnet_encoder': self.pointnet_encoder,
                'center_pred': self.center_pred,
                'center_semantic': self.center_semantic,
                'center_offset': self.center_offset,
            }

        ### only the upper branch of our target model
        elif self.model_mode == 'Yu_refine_clustering_PointGroup':
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, backbone=True, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.point_refine_feature_attn = nn.MultiheadAttention(embed_dim=m, num_heads=cfg.multi_heads)
            self.atten_outputlayer = nn.Sequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.point_offset = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, 3, bias=True),
            )

            self.point_semantic = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, classes, bias=True),
            )

            module_map = {
                'module.input_conv': self.input_conv, 'module.unet': self.unet,
                'module.output_layer': self.output_layer,
                'module.point_refine_feature_attn': self.point_refine_feature_attn,
                'module.atten_outputlayer': self.atten_outputlayer,
            }

        elif self.model_mode == 'Yu_refine_clustering_scorenet_PointGroup':
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, backbone=True, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.point_offset = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, 3, bias=True),
            )

            self.point_semantic = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, classes, bias=True),
            )

            self.point_refine_feature_attn = nn.MultiheadAttention(embed_dim=m, num_heads=cfg.multi_heads)
            self.atten_outputlayer = nn.Sequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.cluster_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1)
            self.cluster_outputlayer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            module_map = {
                'module.input_conv': self.input_conv, 'module.unet': self.unet,
                'module.output_layer': self.output_layer,
                'module.point_refine_feature_attn': self.point_refine_feature_attn,
                'module.atten_outputlayer': self.atten_outputlayer,
                'module.cluster_unet': self.cluster_unet, 'module.cluster_outputlayer': self.cluster_outputlayer,
            }

        elif self.model_mode == 'Yu_stuff_recurrent_PointGroup':
            self.stuff_recurrent = cfg.stuff_recurrent

            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.point_offset = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, 3, bias=True),
            )

            self.point_semantic = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, classes, bias=True),
            )

            self.apply(self.set_bn_init)

            module_map = {
                'module.input_conv': self.input_conv,
                'module.unet': self.unet,
                'module.output_layer': self.output_layer,
            }

        elif self.model_mode == 'Yu_stuff_remove_PointGroup':
            self.stuff_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.stuff_unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, backbone=True)

            self.stuff_output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.stuff_linear = nn.Linear(m, 2)

            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.point_offset = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, 3, bias=True),
            )

            self.point_semantic = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, classes, bias=True),
            )

            self.apply(self.set_bn_init)

            module_map = {
                'module.stuff_conv': self.stuff_conv,
                'module.stuff_unet': self.stuff_unet,
                'module.stuff_output_layer': self.stuff_output_layer,
                'module.stuff_linear': self.stuff_linear,
                'module.input_conv': self.input_conv,
                'module.unet': self.unet,
                'module.output_layer': self.output_layer,
            }

        self.local_pretrained_model_parameter()

    def forward(self, input, input_map, coords, rgb, ori_coords, batch_idxs, batch_offsets, epoch):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        batch_idxs = batch_idxs.squeeze()

        if self.model_mode == 'Center_clustering':
            semantic_scores = []
            point_offset_preds = []

            point_feats, grid_feats = self.pointnet_backbone_forward(coords, coords, rgb, batch_offsets)

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
            semantic_scores.append(self.point_semantic(output_feats))  # (N, nClass), float
            # point_semantic_preds = semantic_scores
            point_semantic_preds = semantic_scores[0].max(1)[1]
            #### point offset prediction
            point_offset_preds.append(self.point_offset(output_feats))  # (N, 3), float32

            ### center prediction
            bs, c_dim, grid_size = grid_feats.shape[0], grid_feats.shape[1], grid_feats.shape[2]
            grid_feats = grid_feats.reshape(bs, c_dim, -1).permute(0, 2, 1)

            center_preds = self.center_pred(grid_feats)
            center_semantic_preds = self.center_semantic(grid_feats)
            center_offset_preds = self.center_offset(grid_feats)

            ret['point_semantic_scores'] = semantic_scores
            ret['point_offset_preds'] = point_offset_preds

            ret['center_preds'] = center_preds
            ret['center_semantic_preds'] = center_semantic_preds
            ret['center_offset_preds'] = center_offset_preds

        elif self.model_mode == 'Center_pointnet++_clustering':
            semantic_scores = []
            point_offset_preds = []

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
            semantic_scores.append(self.point_semantic(output_feats))  # (N, nClass), float
            # point_semantic_preds = semantic_scores
            point_semantic_preds = semantic_scores[0].max(1)[1]
            #### point offset prediction
            point_offset_preds.append(self.point_offset(output_feats))  # (N, 3), float32

            point_feats = []
            sampled_indexes = []
            for sample_indx in range(1, len(batch_offsets)):
                coords_input = coords[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)
                rgb_input = rgb[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)
                if not input['test'] and self.center_clustering['use_gt_semantic']:
                    point_semantic_preds = input['semantic_labels']
                point_semantic_pred = point_semantic_preds[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx]]
                for semantic_idx in point_semantic_pred.unique():
                    if semantic_idx < 2:
                        continue
                    semantic_point_idx = (point_semantic_pred == semantic_idx).nonzero().squeeze(dim=1)

                    sampled_index = pointnet2_utils.furthest_point_sample(
                        coords_input[:, semantic_point_idx, :3].contiguous(), self.pointnet_max_npoint
                    ).squeeze(dim=0).long()
                    sampled_index = semantic_point_idx[sampled_index]
                    sampled_indexes.append(torch.cat(
                        (torch.LongTensor(sampled_index.shape[0], 1).fill_(sample_indx).cuda(),
                         sampled_index.unsqueeze(dim=1)), dim=1)
                    )

                    sampled_coords_input = coords_input[:, sampled_index, :]
                    sampled_rgb_input = rgb_input[:, sampled_index, :]

                    point_feat, _ = self.pointnet_encoder(
                        sampled_coords_input,
                        torch.cat((sampled_rgb_input, sampled_coords_input), dim=2).transpose(1, 2).contiguous()
                    )
                    point_feats.append(point_feat)

            point_feats = torch.cat(point_feats, dim=0)
            sampled_indexes = torch.cat(sampled_indexes, dim=0)

            ### center prediction
            center_preds = self.center_pred(point_feats)
            center_semantic_preds = self.center_semantic(point_feats)
            center_offset_preds = self.center_offset(point_feats)

            ret['point_semantic_scores'] = semantic_scores
            ret['point_offset_preds'] = point_offset_preds

            ret['center_preds'] = (center_preds, sampled_indexes)
            ret['center_semantic_preds'] = (center_semantic_preds, sampled_indexes)
            ret['center_offset_preds'] = (center_offset_preds, sampled_indexes)

            ret['point_features'] = output_feats

        elif self.model_mode == 'Yu_refine_clustering_PointGroup':
            point_offset_preds = []
            point_semantic_scores = []

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
            point_semantic_scores.append(self.point_semantic(output_feats))  # (N, nClass), float
            # point_semantic_preds = semantic_scores
            point_semantic_preds = point_semantic_scores[-1].max(1)[1]

            #### point offset prediction
            point_offset_preds.append(self.point_offset(output_feats))  # (N, 3), float32

            point_features = output_feats.clone()

            if (epoch > self.prepare_epochs):

                for _ in range(self.proposal_refinement['refine_times']):
                    #### get prooposal clusters
                    object_idxs = torch.nonzero(point_semantic_preds > 1).view(-1)

                    batch_idxs_ = batch_idxs[object_idxs]
                    batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input['batch_size'])
                    coords_ = coords[object_idxs]
                    pt_offsets_ = point_offset_preds[-1][object_idxs]

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

                    c_idxs = proposals_idx_shift[:, 1].cuda()
                    clusters_feats = output_feats[c_idxs.long()]

                    cluster_feature = pointgroup_ops.sec_mean(clusters_feats, proposals_offset_shift.cuda())  # (nCluster, m), float

                    clusters_pts_idxs = proposals_idx_shift[proposals_offset_shift[:-1].long()][:, 1].cuda()

                    if len(clusters_pts_idxs) == 0:
                        continue

                    clusters_batch_idxs = clusters_pts_idxs.clone()
                    for _batch_idx in range(len(batch_offsets_) - 1, 0, -1):
                        clusters_batch_idxs[clusters_pts_idxs < batch_offsets_[_batch_idx]] = _batch_idx

                    refined_point_features = []
                    for _batch_idx in range(1, len(batch_offsets)):
                        point_refined_feature, _ = self.point_refine_feature_attn(
                            query=output_feats[batch_offsets[_batch_idx-1]:batch_offsets[_batch_idx], :].unsqueeze(dim=1),
                            key=cluster_feature[clusters_batch_idxs == _batch_idx, :].unsqueeze(dim=1),
                            value=cluster_feature[clusters_batch_idxs == _batch_idx, :].unsqueeze(dim=1)
                        )
                        point_refined_feature = self.atten_outputlayer(point_refined_feature.squeeze(dim=1))
                        refined_point_features.append(point_refined_feature)

                    refined_point_features = torch.cat(refined_point_features, dim=0)
                    assert refined_point_features.shape[0] == point_features.shape[0], 'point wise features have wrong point numbers'

                    refined_point_features = refined_point_features + point_features
                    point_features = refined_point_features.clone()

                    ### refined point prediction
                    #### refined point semantic label prediction
                    point_semantic_scores.append(self.point_semantic(refined_point_features))  # (N, nClass), float
                    point_semantic_preds = point_semantic_scores[-1].max(1)[1]

                    #### point offset prediction
                    point_offset_preds.append(self.point_offset(refined_point_features))  # (N, 3), float32

                if (epoch == self.test_epoch) and input['test']:
                    self.cluster_sets = 'Q'
                    scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                        coords, point_offset_preds[-1], point_semantic_preds,
                        batch_idxs, input['batch_size']
                    )
                    ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = point_semantic_scores
            ret['point_offset_preds'] = point_offset_preds

        elif self.model_mode == 'Fan_center_loss_PointGroup':
            semantic_scores = []
            point_offset_preds = []
            points_semantic_center_loss_feature = []

            voxel_feats = pointgroup_ops.voxelization(input['pt_feats'], input['v2p_map'],
                                                      input['mode'])  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(
                voxel_feats, input['voxel_coords'], input['spatial_shape'], input['batch_size']
            )
            output = self.input_conv(input_)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[input_map.long()]
            output_feats = output_feats.squeeze(dim=0)

            points_semantic_center_loss_feature.append(output_feats)

            ### point prediction
            #### point semantic label prediction
            semantic_scores.append(self.point_deeper_semantic(output_feats))  # (N, nClass), float

            point_semantic_preds = semantic_scores[0].max(1)[1]

            #### point offset prediction
            point_offset_preds.append(self.point_offset(output_feats))  # (N, 3), float32
            # only used to evaluate based on ground truth
            # point_offset_preds.append(input['point_offset_preds'])  # (N, 3), float32

            if (epoch > self.prepare_epochs):
                #### get prooposal clusters
                object_idxs = torch.nonzero(point_semantic_preds > 1).view(-1)

                batch_idxs_ = batch_idxs[object_idxs]
                batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input['batch_size'])
                coords_ = coords[object_idxs]
                pt_offsets_ = point_offset_preds[0][object_idxs]

                semantic_preds_cpu = point_semantic_preds[object_idxs].int().cpu()

                idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(
                    coords_ + pt_offsets_ + (torch.rand(coords_.shape) * 1e-2).cuda(), batch_idxs_,
                    batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive
                )
                # idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(
                #     coords_ + pt_offsets_ + (torch.rand(coords_.shape) * 1e-2).cuda(), batch_idxs_,
                #     batch_offsets_, 0.001, self.cluster_shift_meanActive
                # )
                proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(
                    semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre
                )

                proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
                # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset_shift: (nProposal + 1), int

                idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_,
                                                                  self.cluster_radius,
                                                                  self.cluster_meanActive)
                proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(),
                                                                             start_len.cpu(),
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
            ret['points_semantic_center_loss_feature'] = points_semantic_center_loss_feature

        elif self.model_mode == 'Yu_refine_clustering_scorenet_PointGroup':
            point_offset_preds = []
            point_semantic_scores = []

            voxel_feats = pointgroup_ops.voxelization(input['pt_feats'], input['v2p_map'],
                                                      input['mode'])  # (M, C), float, cuda

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
            point_semantic_scores.append(self.point_semantic(output_feats))  # (N, nClass), float
            # point_semantic_preds = semantic_scores
            point_semantic_preds = point_semantic_scores[-1].max(1)[1]

            #### point offset prediction
            point_offset_preds.append(self.point_offset(output_feats))  # (N, 3), float32

            point_features = output_feats.clone()

            if (epoch > self.prepare_epochs):

                for _ in range(self.proposal_refinement['refine_times']):
                    #### get prooposal clusters
                    object_idxs = torch.nonzero(point_semantic_preds > 1).view(-1)

                    batch_idxs_ = batch_idxs[object_idxs]
                    batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input['batch_size'])
                    coords_ = coords[object_idxs]
                    pt_offsets_ = point_offset_preds[-1][object_idxs]

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

                    if proposals_idx_shift.shape[0] == 0:
                        continue

                    #### proposals voxelization again
                    input_feats, inp_map = self.clusters_voxelization(
                        proposals_idx_shift, proposals_offset_shift, output_feats, coords,
                        self.score_fullscale, self.score_scale, self.mode
                    )

                    #### cluster features
                    clusters = self.cluster_unet(input_feats)
                    clusters = self.cluster_outputlayer(clusters)
                    cluster_feature = clusters.features[inp_map.long()]  # (sumNPoint, C)
                    cluster_feature = pointgroup_ops.roipool(cluster_feature, proposals_offset_shift.cuda())  # (nProposal, C)

                    clusters_pts_idxs = proposals_idx_shift[proposals_offset_shift[:-1].long()][:, 1].cuda()

                    if len(clusters_pts_idxs) == 0:
                        continue

                    clusters_batch_idxs = clusters_pts_idxs.clone()
                    for _batch_idx in range(len(batch_offsets_) - 1, 0, -1):
                        clusters_batch_idxs[clusters_pts_idxs < batch_offsets_[_batch_idx]] = _batch_idx

                    refined_point_features = []
                    for _batch_idx in range(1, len(batch_offsets)):
                        point_refined_feature, _ = self.point_refine_feature_attn(
                            query=output_feats[batch_offsets[_batch_idx - 1]:batch_offsets[_batch_idx], :].unsqueeze(
                                dim=1),
                            key=cluster_feature[clusters_batch_idxs == _batch_idx, :].unsqueeze(dim=1),
                            value=cluster_feature[clusters_batch_idxs == _batch_idx, :].unsqueeze(dim=1)
                        )
                        point_refined_feature = self.atten_outputlayer(point_refined_feature.squeeze(dim=1))
                        refined_point_features.append(point_refined_feature)

                    refined_point_features = torch.cat(refined_point_features, dim=0)
                    assert refined_point_features.shape[0] == point_features.shape[
                        0], 'point wise features have wrong point numbers'

                    refined_point_features = refined_point_features + point_features
                    point_features = refined_point_features.clone()

                    ### refined point prediction
                    #### refined point semantic label prediction
                    point_semantic_scores.append(self.point_semantic(refined_point_features))  # (N, nClass), float
                    point_semantic_preds = point_semantic_scores[-1].max(1)[1]

                    #### point offset prediction
                    point_offset_preds.append(self.point_offset(refined_point_features))  # (N, 3), float32

                if (epoch == self.test_epoch) and input['test']:
                    self.cluster_sets = 'Q'
                    scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                        coords, point_offset_preds[-1], point_semantic_preds,
                        batch_idxs, input['batch_size']
                    )
                    ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = point_semantic_scores
            ret['point_offset_preds'] = point_offset_preds

        elif self.model_mode == 'Yu_stuff_recurrent_PointGroup':
            point_offset_preds = []
            point_semantic_scores = []

            point_semantic_recurrent_scores = []
            point_offset_recurrent_preds = []

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
            point_semantic_scores.append(self.point_semantic(output_feats))  # (N, nClass), float
            point_semantic_preds = point_semantic_scores[-1].max(1)[1]

            #### point offset prediction
            point_offset_preds.append(self.point_offset(output_feats))  # (N, 3), float32

            point_semantic_prediction = point_semantic_preds
            for _ in range(self.stuff_recurrent['recurrent_times']):
                point_semantic_prediction[point_semantic_preds < 2] = 0

                non_stuff_index = point_semantic_prediction > 1

                nonstuff_voxel_locs, nonstuff_p2v_map, nonstuff_v2p_map = pointgroup_ops.voxelization_idx(
                    input['point_locs'][non_stuff_index].cpu().clone(), self.batch_size, self.mode
                )

                nonstuff_voxel_locs = nonstuff_voxel_locs.int().cuda()
                nonstuff_v2p_map = nonstuff_v2p_map.cuda()

                nonstuff_voxel_feats = pointgroup_ops.voxelization(
                    input['pt_feats'][non_stuff_index], nonstuff_v2p_map,
                    input['mode']
                )  # (M, C), float, cuda

                nonstuff_spatial_shape = np.clip(
                    (input['point_locs'][non_stuff_index].max(0)[0][1:] + 1).cpu().numpy(),
                    self.full_scale[0], None
                )  # long (3)
                nonstuff_input_map = nonstuff_p2v_map.cuda()

                input_ = spconv.SparseConvTensor(
                    nonstuff_voxel_feats, nonstuff_voxel_locs, nonstuff_spatial_shape, input['batch_size']
                )

                output = self.input_conv(input_)
                output = self.unet(output)
                output = self.output_layer(output)
                output_feats = output.features[nonstuff_input_map.long()]
                output_feats = output_feats.squeeze(dim=0)

                point_semantic_recurrent_scores.append((self.point_semantic(output_feats), non_stuff_index))  # (N, nClass), float
                point_semantic_preds = point_semantic_recurrent_scores[-1][0].max(1)[1]

                point_offset_recurrent_preds.append((self.point_offset(output_feats), non_stuff_index))  # (N, 3), float32

            if (epoch == self.test_epoch) and input['test']:
                self.cluster_sets = 'Q'

                nonstuff_point_semantic_preds = point_semantic_recurrent_scores[-1][0].max(1)[1]
                point_semantic_pred_full = torch.zeros(coords.shape[0], dtype=torch.long).cuda()
                point_semantic_pred_full[
                    (point_semantic_preds > 1).nonzero().squeeze(dim=1).long()] = nonstuff_point_semantic_preds[
                    (point_semantic_preds > 1).nonzero().squeeze(dim=1).long()]

                point_offset_pred = torch.zeros((coords.shape[0], 3), dtype=torch.float).cuda()
                point_offset_pred[(point_semantic_preds > 1).nonzero().squeeze(dim=1).long()] = point_offset_preds[-1][
                    (point_semantic_preds > 1).nonzero().squeeze(dim=1).long()]

                # TODO: need to change stuff_preds
                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_pred, point_semantic_pred_full,
                    batch_idxs, input['batch_size'], stuff_preds=non_stuff_index
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)
                ret['point_semantic_pred_full'] = point_semantic_pred_full

            ret['point_semantic_scores'] = point_semantic_scores
            ret['point_offset_preds'] = point_offset_preds

        elif self.model_mode == 'Yu_stuff_remove_PointGroup':
            point_offset_preds = []
            point_semantic_scores = []

            voxel_stuff_feats = pointgroup_ops.voxelization(input['pt_feats'], input['v2p_map'], input['mode'])  # (M, C), float, cuda

            stuff_input_ = spconv.SparseConvTensor(
                voxel_stuff_feats, input['voxel_coords'],
                input['spatial_shape'], input['batch_size']
            )

            stuff_output = self.stuff_conv(stuff_input_)
            stuff_output = self.stuff_unet(stuff_output)
            stuff_output = self.stuff_output_layer(stuff_output)
            stuff_output_feats = stuff_output.features[input_map.long()]
            stuff_output_feats = stuff_output_feats.squeeze(dim=0)

            stuff_preds = self.stuff_linear(stuff_output_feats)

            if 'nonstuff_feats' in input.keys():
                nonstuff_voxel_feats = pointgroup_ops.voxelization(
                    input['nonstuff_feats'], input['nonstuff_v2p_map'],
                    input['mode']
                )  # (M, C), float, cuda

                nonstuff_voxel_locs = input['nonstuff_voxel_locs']
                nonstuff_spatial_shape = input['nonstuff_spatial_shape']
                nonstuff_input_map = input['nonstuff_p2v_map']
            else:
                nonstuff_voxel_locs, nonstuff_p2v_map, nonstuff_v2p_map = pointgroup_ops.voxelization_idx(
                    input['point_locs'][stuff_preds.max(1)[1] == 1].cpu().clone(), self.batch_size, self.mode
                )

                nonstuff_voxel_locs = nonstuff_voxel_locs.int().cuda()
                nonstuff_v2p_map = nonstuff_v2p_map.cuda()

                nonstuff_voxel_feats = pointgroup_ops.voxelization(
                    input['pt_feats'][stuff_preds.max(1)[1] == 1], nonstuff_v2p_map,
                    input['mode']
                )  # (M, C), float, cuda

                nonstuff_spatial_shape = np.clip(
                    (input['point_locs'][stuff_preds.max(1)[1] == 1].max(0)[0][1:] + 1).cpu().numpy(),
                    self.full_scale[0], None
                )  # long (3)
                nonstuff_input_map = nonstuff_p2v_map.cuda()

            input_ = spconv.SparseConvTensor(
                nonstuff_voxel_feats, nonstuff_voxel_locs, nonstuff_spatial_shape, input['batch_size']
            )

            output = self.input_conv(input_)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[nonstuff_input_map.long()]
            output_feats = output_feats.squeeze(dim=0)

            ### point prediction
            #### point semantic label prediction
            point_semantic_scores.append(self.point_semantic(output_feats))  # (N, nClass), float
            # point_semantic_preds = semantic_scores

            #### point offset prediction
            nonstuff_point_offset_pred = self.point_offset(output_feats)
            point_offset_preds.append(nonstuff_point_offset_pred)  # (N, 3), float32

            if (epoch == self.test_epoch) and input['test']:
                self.cluster_sets = 'Q'

                nonstuff_point_semantic_preds = point_semantic_scores[-1].max(1)[1] + 2
                point_semantic_pred_full = torch.zeros(coords.shape[0], dtype=torch.long).cuda()
                point_semantic_pred_full[
                    (stuff_preds.max(1)[1] == 1).nonzero().squeeze(dim=1).long()] = nonstuff_point_semantic_preds[
                    (stuff_preds.max(1)[1] == 1).nonzero().squeeze(dim=1).long()]

                point_offset_pred = torch.zeros((coords.shape[0], 3), dtype=torch.float).cuda()
                point_offset_pred[(stuff_preds.max(1)[1] == 1).nonzero().squeeze(dim=1).long()] = nonstuff_point_offset_pred[
                    (stuff_preds.max(1)[1] == 1).nonzero().squeeze(dim=1).long()]

                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_pred, point_semantic_pred_full,
                    batch_idxs, input['batch_size'], stuff_preds=stuff_preds.max(1)[1]
                )

                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)
                ret['point_semantic_pred_full'] = point_semantic_pred_full

            ret['point_semantic_scores'] = point_semantic_scores
            ret['point_offset_preds'] = point_offset_preds
            ret['stuff_preds'] = stuff_preds
            ret['output_feats'] = stuff_output_feats


        return ret
