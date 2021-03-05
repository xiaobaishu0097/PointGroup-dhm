import torch
import torch.nn as nn
import spconv

from lib.pointgroup_ops.functions import pointgroup_ops
from model.encoder import pointnet, pointnetpp
from model.encoder.unet3d import UNet3D
from model.decoder import decoder
from model.components import UBlock, backbone_pointnet2
from model.Pointnet2.pointnet2 import pointnet2_utils
from model.basemodel import BaseModel


class CenterSemanticSampled(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.center_clustering = cfg.center_clustering

        '''point-wise prediction networks'''
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

        '''center prediction network'''
        self.pointnet_encoder = backbone_pointnet2(output_dim=self.m)

        self.center_pred = nn.Sequential(
            nn.Linear(self.m, self.m),
            nn.ReLU(),
            nn.Linear(self.m, 1)
        )

        self.center_semantic = nn.Sequential(
            nn.Linear(self.m, self.m),
            nn.ReLU(),
            nn.Linear(self.m, self.classes)
        )

        self.center_offset = nn.Sequential(
            nn.Linear(self.m, self.m),
            nn.ReLU(),
            nn.Linear(self.m, 3)
        )

        ### center prediction branch
        ### convolutional occupancy networks decoder
        self.center_decoder = decoder.LocalDecoder(
            dim=3,
            c_dim=32,
            hidden_size=32,
        )

        self.module_map = {
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

        self.local_pretrained_model_parameter()

    def forward(self, input, input_map, coords, rgb, ori_coords, batch_idxs, batch_offsets, epoch):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        semantic_scores = []
        point_offset_preds = []

        voxel_feats = pointgroup_ops.voxelization(
            input['pt_feats'], input['v2p_map'], input['mode'])  # (M, C), float, cuda

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

        return ret
