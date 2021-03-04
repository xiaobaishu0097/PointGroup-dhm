import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import spconv

from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
from model.components import UBlock, euclidean_dist
from model.basemodel import BaseModel


class LocalProposalRefinement(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.local_proposal = cfg.local_proposal

        '''backbone network'''
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

        '''point-wise prediction networks'''
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

        self.module_map = {
            'input_conv': self.input_conv,
            'unet': self.unet,
            'output_layer': self.output_layer,
            'point_offset': self.point_offset,
            'point_semantic': self.point_semantic,
        }

        '''local proposal refinement network'''
        if not self.local_proposal['reuse_backbone_unet']:
            self.proposal_unet = UBlock(
                [self.m, 2 * self.m, 3 * self.m, 4 * self.m, 5 * self.m, 6 * self.m, 7 * self.m],
                self.norm_fn, self.block_reps, self.block, indice_key_id=1, backbone=False
            )
            
            self.proposal_outputlayer = spconv.SparseSequential(
                self.norm_fn(self.m),
                nn.ReLU()
            )

            self.module_map['proposal_unet'] = self.proposal_unet
            self.module_map['proposal_outputlayer'] = self.proposal_outputlayer

        self.local_pretrained_model_parameter()

    def forward(self, input, input_map, coords, rgb, ori_coords, batch_idxs, batch_offsets, epoch):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        point_semantic_scores = []
        point_offset_preds = []
        local_point_semantic_scores = []
        local_point_offset_preds = []

        '''point feature extraction'''
        # voxelize point cloud, and those voxels are passed to a sparse 3D U-Net
        # the output of sparse 3D U-Net is then remapped back to points
        voxel_feats = pointgroup_ops.voxelization(
            input['pt_feats'], input['v2p_map'], input['mode']
        )  # (M, C), float, cuda
        input_ = spconv.SparseConvTensor(
            voxel_feats, input['voxel_coords'], input['spatial_shape'], input['batch_size']
        )
        output = self.input_conv(input_)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        '''point-wise predictions'''
        point_semantic_scores.append(self.point_semantic(output_feats))  # (N, nClass), float
        point_semantic_preds = point_semantic_scores[-1].max(1)[1]

        point_offset_preds.append(self.point_offset(output_feats))  # (N, 3), float32

        # cluster proposals with the stable output of the backbone network
        if (epoch > self.prepare_epochs):
            if not input['test'] and self.local_proposal['use_gt_semantic']:
                point_semantic_preds = input['semantic_labels']

            '''clustering algorithm'''
            object_idxs = torch.nonzero(point_semantic_preds > 1).view(-1)
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input['batch_size'])
            coords_ = coords[object_idxs]
            pt_offsets_ = point_offset_preds[-1][object_idxs]

            semantic_preds_cpu = point_semantic_preds[object_idxs].int().cpu()

            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(
                coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive
            )
            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(
                semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre
            )
            proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
            # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset_shift: (nProposal + 1), int

            '''local proposal refinement'''
            local_proposals_idx = []
            local_proposals_offset = [0]

            # compute proposal centers
            proposal_center_coords = []
            for proposal_index in range(1, len(proposals_offset_shift)):
                proposal_point_index = proposals_idx_shift[
                                 proposals_offset_shift[proposal_index - 1]: proposals_offset_shift[proposal_index], 1]
                proposal_center_coords.append(coords[proposal_point_index.long(), :].mean(dim=0, keepdim=True))
            proposal_center_coords = torch.cat(proposal_center_coords, dim=0)

            # select the topk closest proposals for each proposal
            proposal_dist_mat = euclidean_dist(proposal_center_coords, proposal_center_coords)
            proposal_dist_mat[range(len(proposal_dist_mat)), range(len(proposal_dist_mat))] = 100
            closest_proposals_dist, closest_proposals_index = proposal_dist_mat.topk(
                k=min(self.local_proposal['topk'], proposal_dist_mat.shape[1]), dim=1, largest=False
            )
            valid_closest_proposals_index = closest_proposals_dist < self.local_proposal['dist_th']

            # select proposals which are the closest and in the distance threshold
            for proposal_index in range(1, len(proposals_offset_shift)):
                local_indexs = []
                local_indexs.append(
                    proposals_idx_shift[proposals_offset_shift[proposal_index - 1]:
                                        proposals_offset_shift[proposal_index]])
                closest_proposals_ind = closest_proposals_index[proposal_index - 1, :]
                for selected_proposal_index in closest_proposals_ind[
                    valid_closest_proposals_index[proposal_index - 1, :]]:
                    local_index = proposals_idx_shift[
                                  proposals_offset_shift[selected_proposal_index]:proposals_offset_shift[
                                      selected_proposal_index + 1]][:, 1].unsqueeze(dim=1)
                    local_indexs.append(torch.cat(
                        (torch.LongTensor(local_index.shape[0], 1).fill_(proposal_index - 1).int(),
                         local_index.cpu()), dim=1
                    ))

                local_proposals_idx.append(torch.cat(local_indexs, dim=0))
                local_proposals_offset.append(local_proposals_offset[-1] + local_proposals_idx[-1].shape[0])

            local_proposals_idx = torch.cat(local_proposals_idx, dim=0)
            local_proposals_offset = torch.tensor(local_proposals_offset).int()

            #### proposals voxelization again
            input_feats, inp_map = self.clusters_voxelization(
                local_proposals_idx, local_proposals_offset, output_feats,
                coords, self.local_proposal['local_proposal_full_scale'],
                self.local_proposal['local_proposal_scale'], self.mode
            )

            #### cluster features
            if self.local_proposal['reuse_backbone_unet']:
                proposals = self.unet(input_feats)
                proposals = self.output_layer(proposals)
            else:
                proposals = self.proposal_unet(input_feats)
                proposals = self.proposal_outputlayer(proposals)
            proposals_point_features = proposals.features[inp_map.long()]  # (sumNPoint, C)

            ret['proposals_point_features'] = (proposals_point_features, local_proposals_idx)

            ### scatter mean point predictions
            if self.local_proposal['scatter_mean_target'] == 'prediction':
                refined_point_semantic_score = point_semantic_scores[-1]
                local_point_semantic_score = self.point_semantic(proposals_point_features)
                refined_point_semantic_score[:min((local_proposals_idx[:, 1].max() + 1).item(), coords.shape[0]), :] = \
                    scatter_mean(local_point_semantic_score, local_proposals_idx[:, 1].cuda().long(), dim=0)
                point_semantic_scores.append(refined_point_semantic_score)
                point_semantic_preds = refined_point_semantic_score.max(1)[1]

                refined_point_offset_pred = point_offset_preds[-1]
                local_point_offset_pred = self.point_offset(proposals_point_features)
                refined_point_offset_pred[:min((local_proposals_idx[:, 1].max() + 1).item(), coords.shape[0]), :] = \
                    scatter_mean(local_point_offset_pred, local_proposals_idx[:, 1].cuda().long(), dim=0)
                point_offset_preds.append(refined_point_offset_pred)

            ### scatter mean point features
            elif self.local_proposal['scatter_mean_target'] == 'feature':
                refined_point_features = torch.zeros_like(output_feats).cuda()
                refined_point_features[:min((local_proposals_idx[:, 1].max() + 1).item(), coords.shape[0]), :] = \
                    scatter_mean(proposals_point_features, local_proposals_idx[:, 1].cuda().long(), dim=0)
                #### filling 0 rows with output_feats
                filled_index = (refined_point_features == torch.zeros_like(refined_point_features[0, :])).all(dim=1)
                refined_point_features[filled_index, :] = output_feats[filled_index, :]

                ### refined point prediction
                #### refined point semantic label prediction
                point_semantic_scores.append(self.point_semantic(refined_point_features))  # (N, nClass), float
                point_semantic_preds = point_semantic_scores[-1].max(1)[1]

                #### point offset prediction
                point_offset_preds.append(self.point_offset(output_feats + refined_point_features))  # (N, 3), float32

            elif self.local_proposal['scatter_mean_target'] == False:
                local_point_semantic_scores.append(self.point_semantic(proposals_point_features))
                local_point_offset_preds.append(self.point_offset(proposals_point_features))

                ret['local_point_semantic_scores'] = (local_point_semantic_scores, local_proposals_idx)
                ret['local_point_offset_preds'] = (local_point_offset_preds, local_proposals_idx)

            if input['test']:
                if self.local_proposal['scatter_mean_target'] == False:
                    local_point_semantic_score = self.point_semantic(proposals_point_features)
                    local_point_offset_pred = self.point_offset(proposals_point_features)

                    local_point_semantic_score = scatter_mean(
                        local_point_semantic_score, local_proposals_idx[:, 1].cuda().long(), dim=0)

                    point_semantic_score = point_semantic_scores[-1]
                    point_semantic_score[:local_point_semantic_score.shape[0], :] += local_point_semantic_score
                    point_semantic_preds = point_semantic_score.max(1)[1]

                    point_offset_pred = point_offset_preds[-1]
                    local_point_offset_pred = scatter_mean(
                        local_point_offset_pred, local_proposals_idx[:, 1].cuda().long(), dim=0)
                    point_offset_pred[:local_point_offset_pred.shape[0], :] = local_point_offset_pred

                self.cluster_sets = 'Q'
                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_pred, point_semantic_preds, batch_idxs, input['batch_size']
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

        ret['point_semantic_scores'] = point_semantic_scores
        ret['point_offset_preds'] = point_offset_preds
        ret['point_features'] = output_feats

        return ret
