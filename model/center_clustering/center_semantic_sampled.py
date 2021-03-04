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
        if self.backbone == 'pointnet':
            #### PointNet backbone encoder
            self.pointnet_encoder = pointnet.LocalPoolPointnet(
                c_dim=self.m, dim=6, hidden_dim=self.m, scatter_type=cfg.scatter_type, grid_resolution=32,
                plane_type='grid', padding=0.1, n_blocks=5
            )

        elif self.backbone == 'pointnet++_yanx':
            self.pointnet_encoder = pointnetpp.PointNetPlusPlus(
                c_dim=self.m, include_rgb=self.pointnet_include_rgb
            )

        elif self.backbone == 'pointnet++_shi':
            self.pointnet_encoder = backbone_pointnet2(output_dim=self.m)

        self.unet3d = UNet3D(
            num_levels=cfg.unet3d_num_levels, f_maps=self.m, in_channels=self.m, out_channels=self.m
        )

        self.decoder = decoder.LocalDecoder(
            dim=3,
            c_dim=32,
            hidden_size=32,
        )

        ### point prediction branch
        ### sparse 3D U-Net
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

        module_map = {}

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

        ### lower branch --> grid-wise predictions
        point_feats = []
        grid_feats = []

        queries_feats = []

        for sample_indx in range(1, len(batch_offsets)):
            coords_input = coords[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)
            rgb_input = rgb[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)

            sampled_index = pointnet2_utils.furthest_point_sample(
                coords_input[:, :, :3].contiguous(), self.pointnet_max_npoint).squeeze(dim=0).long()

            sampled_coords_input = coords_input[:, sampled_index, :]
            sampled_rgb_input = rgb_input[:, sampled_index, :]

            point_feat, _ = self.pointnet_encoder(
                sampled_coords_input,
                torch.cat((sampled_rgb_input, sampled_coords_input), dim=2).transpose(1, 2).contiguous()
            )
            grid_feats.append(self.generate_grid_features(sampled_coords_input, point_feat))

            if not input['test']:
                decoder_input = {'grid': grid_feats[-1]}
                center_queries_coords_input = input['center_queries_coords'][
                                              input['center_queries_batch_offsets'][sample_indx - 1]:
                                              input['center_queries_batch_offsets'][sample_indx], :].unsqueeze(dim=0)
                queries_feats.append(self.decoder(center_queries_coords_input, decoder_input).squeeze(dim=0))

        grid_feats = torch.cat(grid_feats, dim=0).contiguous()
        if not input['test']:
            queries_feats = torch.cat(queries_feats, dim=0).contiguous()

        ### upper branch --> point-wise predictions
        voxel_feats = pointgroup_ops.voxelization(
            input['pt_feats'], input['v2p_map'], input['mode'])  # (M, C), float, cuda

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

        #### point offset prediction
        point_offset_preds.append(self.point_offset(output_feats))  # (N, 3), float32

        ### center prediction
        bs, c_dim, grid_size = grid_feats.shape[0], grid_feats.shape[1], grid_feats.shape[2]
        grid_feats = grid_feats.reshape(bs, c_dim, -1).permute(0, 2, 1)

        center_preds = self.center_pred(grid_feats)
        center_semantic_preds = self.center_semantic(grid_feats)
        center_offset_preds = self.center_offset(grid_feats)

        if not input['test']:
            queries_preds = self.center_pred(queries_feats)
            queries_semantic_preds = self.center_semantic(queries_feats)
            queries_offset_preds = self.center_offset(queries_feats)

        ret['point_semantic_scores'] = semantic_scores
        ret['point_offset_preds'] = point_offset_preds

        ret['center_preds'] = center_preds
        ret['center_semantic_preds'] = center_semantic_preds
        ret['center_offset_preds'] = center_offset_preds

        if not input['test']:
            ret['queries_preds'] = queries_preds
            ret['queries_semantic_preds'] = queries_semantic_preds
            ret['queries_offset_preds'] = queries_offset_preds

        return ret
