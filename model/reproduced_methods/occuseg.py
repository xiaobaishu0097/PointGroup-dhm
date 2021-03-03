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
from model.basemodel import BaseModel


class OccuSeg(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.occupancy_cluster = cfg.occupancy_cluster

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_c, self.m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock(
            [self.m, 2 * self.m, 3 * self.m, 4 * self.m, 5 * self.m, 6 * self.m, 7 * self.m], self.norm_fn,
            self.block_reps, self.block, indice_key_id=1, backbone=True, UNet_Transformer=cfg.UNet_Transformer
        )

        self.output_layer = spconv.SparseSequential(
            self.norm_fn(self.m),
            nn.ReLU()
        )

        self.point_offset = nn.Sequential(
            nn.Linear(self.m, self.m, bias=True),
            self.norm_fn(self.m),
            nn.ReLU(),
            nn.Linear(self.m, 3, bias=True),
        )

        self.point_semantic = nn.Sequential(
            nn.Linear(self.m, self.m, bias=True),
            self.norm_fn(self.m),
            nn.ReLU(),
            nn.Linear(self.m, self.classes, bias=True),
        )

        self.point_occupancy = nn.Sequential(
            nn.Linear(self.m, self.m, bias=True),
            self.norm_fn(self.m),
            nn.ReLU(),
            nn.Linear(self.m, 1, bias=True),
        )

        #### score branch
        self.score_unet = UBlock([self.m, 2 * self.m], self.norm_fn, 2, self.block, indice_key_id=1, backbone=False)
        self.score_outputlayer = spconv.SparseSequential(
            self.norm_fn(self.m),
            nn.ReLU()
        )
        self.score_linear = nn.Linear(self.m, 1)

        self.apply(self.set_bn_init)

        self.module_map = {
            'input_conv': self.input_conv,
            'unet': self.unet,
            'output_layer': self.output_layer,
            'score_unet': self.score_unet,
            'score_outputlayer': self.score_outputlayer,
            'score_linear': self.score_linear,
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

        semantic_scores = []
        point_offset_preds = []
        voxel_occupancy_preds = []

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
        semantic_scores.append(self.point_semantic(output_feats))  # (N, nClass), float

        ### only used to evaluate based on ground truth
        # semantic_scores.append(input['point_semantic_scores'][0])  # (N, nClass), float
        ### ground truth for each category
        # CATE_NUM = 0
        # semantic_output = self.point_semantic(output_feats)
        # if (input['point_semantic_scores'][0].max(dim=1)[1] == CATE_NUM).sum() > 0:
        #     semantic_output[input['point_semantic_scores'][0].max(dim=1)[1] == CATE_NUM] = \
        #     input['point_semantic_scores'][0][input['point_semantic_scores'][0].max(dim=1)[1] == CATE_NUM].float()
            # semantic_output[semantic_output.max(dim=1)[1] == CATE_NUM] = \
            # input['point_semantic_scores'][0][semantic_output.max(dim=1)[1] == CATE_NUM].float()
        # semantic_scores.append(semantic_output)

        point_semantic_preds = semantic_scores[0].max(1)[1]

        #### point offset prediction
        point_offset_pred = self.point_offset(output_feats)
        point_offset_preds.append(point_offset_pred)  # (N, 3), float32
        # only used to evaluate based on ground truth
        # point_offset_preds.append(input['point_offset_preds'])  # (N, 3), float32

        voxel_occupancy_preds.append(self.point_occupancy(output.features))
        point_occupancy_pred = voxel_occupancy_preds[0][input_map.long()].squeeze(dim=1)

        if (epoch > self.prepare_epochs):
            #### get prooposal clusters
            object_idxs = torch.nonzero(point_semantic_preds > 1).view(-1)

            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input['batch_size'])
            coords_ = coords[object_idxs]
            pt_offsets_ = point_offset_preds[0][object_idxs]
            point_occupancy_pred_ = point_occupancy_pred[object_idxs]

            semantic_preds_cpu = point_semantic_preds[object_idxs].int().cpu()

            # idx_occupancy, start_len_occupancy = pointgroup_ops.ballquery_batch_p(
            #     coords_ + pt_offsets_, batch_idxs_,
            #     batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive
            # )
            # proposals_idx_occupancy, proposals_offset_occupancy = pointgroup_ops.bfs_occupancy_cluster(
            #     semantic_preds_cpu, point_occupancy_pred_.cpu(), idx_occupancy.cpu(),
            #     start_len_occupancy.cpu(), self.cluster_npoint_thre, self.occupancy_cluster['occupancy_threshold_shift']
            # )
            # proposals_idx_occupancy[:, 1] = object_idxs[proposals_idx_occupancy[:, 1].long()].int()
            # # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # # proposals_offset_shift: (nProposal + 1), int
            #
            # idx, start_len = pointgroup_ops.ballquery_batch_p(
            #     coords_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive
            # )
            # proposals_idx, proposals_offset = pointgroup_ops.bfs_occupancy_cluster(
            #     semantic_preds_cpu, point_occupancy_pred_.cpu(), idx.cpu(),
            #     start_len.cpu(), self.cluster_npoint_thre, self.occupancy_cluster['occupancy_threshold']
            # )
            # proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()


            idx, start_len = pointgroup_ops.ballquery_batch_p(
                coords_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive
            )
            proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(
                semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre
            )
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()


            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(
                coords_ + pt_offsets_, batch_idxs_,
                batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive
            )
            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(
                semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre
            )
            proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
            # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset_shift: (nProposal + 1), int

            # proposals_idx_occupancy[:, 0] += (proposals_offset.size(0) - 1)
            # proposals_offset_occupancy += proposals_offset[-1]
            # proposals_idx = torch.cat((proposals_idx, proposals_idx_occupancy), dim=0)
            # proposals_offset = torch.cat((proposals_offset, proposals_offset_occupancy[1:]))

            # proposals_idx_filtered = []
            # proposals_offset_filtered = [0]
            # for proposal in proposals_idx_shift[:, 0].unique():
            #     proposal_index = proposals_idx_shift[:, 0] == proposal
            #     proposal_idxs = proposals_idx_shift[proposal_index, 1].long()
            #     proposals_indexs = proposals_idx_shift[proposal_index, :]
            #
            #     proposal_occupancy_mean = point_occupancy_pred[proposal_idxs].mean()
            #
            #     valid_point_index = torch.ones_like(proposal_idxs).byte()
            #     valid_point_index[point_occupancy_pred[proposal_idxs] < proposal_occupancy_mean *
            #                       (1 - self.occupancy_cluster['occupancy_filter_threshold'])] = 0
            #     valid_point_index[point_occupancy_pred[proposal_idxs] > proposal_occupancy_mean *
            #                       (1 + self.occupancy_cluster['occupancy_filter_threshold'])] = 0
            #     proposal_idx_filtered = proposals_indexs[valid_point_index, :]
            #
            #     proposals_idx_filtered.append(proposal_idx_filtered)
            #     proposals_offset_filtered.append(proposals_offset_filtered[-1] + proposal_idx_filtered.shape[0])
            #
            # proposals_idx_filtered = torch.cat(proposals_idx_filtered, dim=0)
            # proposals_offset_filtered = torch.tensor(proposals_offset_filtered).int()
            # proposals_idx_shift = proposals_idx_filtered
            # proposals_offset_shift = proposals_offset_filtered

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

            # proposals_idx = proposals_idx_occupancy
            # proposals_offset = proposals_offset_occupancy
            # scores = torch.ones(proposals_offset.shape[0] - 1, 1).to(point_offset_preds[0].device)

            ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

        ret['point_semantic_scores'] = semantic_scores
        ret['point_offset_preds'] = point_offset_preds
        ret['point_features'] = output_feats
        ret['voxel_occupancy_preds'] = voxel_occupancy_preds

        return ret
