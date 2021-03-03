import torch
import torch.nn as nn
import spconv
import sys

sys.path.append('../')

from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
from model.components import UBlock, ProposalTransformer
from model.basemodel import BaseModel


class ProposalTransformerRefinement(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

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

        self.proposal_transformer = ProposalTransformer(
            d_model=self.m,
            nhead=cfg.Proposal_Transformer['multi_heads'],
            num_decoder_layers=cfg.Proposal_Transformer['num_decoder_layers'],
            dim_feedforward=cfg.Proposal_Transformer['dim_feedforward'],
            dropout=0.0,
        )

        self.proposal_unet = UBlock(
            [self.m, 2 * self.m, 3 * self.m, 4 * self.m], self.norm_fn, 2,
            self.block, indice_key_id=1, backbone=False
        )
        self.proposal_outputlayer = spconv.SparseSequential(
            self.norm_fn(self.m),
            nn.ReLU()
        )
        self.proposal_confidence_linear = nn.Linear(self.m, 1)

        self.module_map = {
            'input_conv': self.input_conv,
            'unet': self.unet,
            'output_layer': self.output_layer,
            'proposal_transformer': self.proposal_transformer,
            'proposal_unet': self.proposal_unet,
            'proposal_outputlayer': self.proposal_outputlayer,
            'proposal_confidence_linear': self.proposal_confidence_linear,
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

        point_offset_preds = []
        point_semantic_scores = []

        proposals_confidence_preds = []
        proposals_idx_shifts = []
        proposals_offset_shifts = []

        voxel_feats = pointgroup_ops.voxelization(input['pt_feats'], input['v2p_map'], input['mode']) # (M, C), float, cuda

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
                    proposals_idx_shift, proposals_offset_shift, output_feats,
                    coords, self.proposal_refinement['proposal_refine_full_scale'],
                    self.proposal_refinement['proposal_refine_scale'], self.mode
                )

                #### proposal features
                proposals = self.proposal_unet(input_feats)
                proposals = self.proposal_outputlayer(proposals)
                proposal_feats = proposals.features[inp_map.long()]  # (sumNPoint, C)
                proposal_feats = pointgroup_ops.roipool(proposal_feats, proposals_offset_shift.cuda())  # (nProposal, C)

                proposals_confidence_preds.append(self.proposal_confidence_linear(proposal_feats))  # (nProposal, 1)
                proposals_idx_shifts.append(proposals_idx_shift)
                proposals_offset_shifts.append(proposals_offset_shift)

                proposal_pts_idxs = proposals_idx_shift[proposals_offset_shift[:-1].long()][:, 1].cuda()

                if len(proposal_pts_idxs) == 0:
                    continue

                proposal_batch_idxs = proposal_pts_idxs.clone()
                for _batch_idx in range(len(batch_offsets_) - 1, 0, -1):
                    proposal_batch_idxs[proposal_pts_idxs < batch_offsets_[_batch_idx]] = _batch_idx

                refined_point_features = []
                for _batch_idx in range(1, len(batch_offsets)):
                    key_input = proposal_feats[proposal_batch_idxs == _batch_idx, :].unsqueeze(dim=0)
                    query_input = output_feats[batch_offsets[_batch_idx - 1]:batch_offsets[_batch_idx],
                                  :].unsqueeze(dim=0).permute(0, 2, 1)
                    # query_input = output_feats[batch_offsets[_batch_idx - 1]:batch_offsets[_batch_idx-1]+100,
                    #               :].unsqueeze(dim=0).permute(0, 2, 1)
                    point_refined_feature, _ = self.proposal_transformer(
                        src=key_input,
                        query_embed=query_input,
                    )

                    refined_point_features.append(point_refined_feature.squeeze(dim=0))

                refined_point_features = torch.cat(refined_point_features, dim=0)
                assert refined_point_features.shape[0] == point_features.shape[
                    0], 'point wise features have wrong point numbers'

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

            ret['proposal_confidences'] = (proposals_confidence_preds, proposals_idx_shifts, proposals_offset_shifts)

        ret['point_semantic_scores'] = point_semantic_scores
        ret['point_offset_preds'] = point_offset_preds

        return ret
