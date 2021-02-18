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

        self.pointnet_max_npoint = 8196

        self.full_scale = cfg.full_scale
        self.batch_size = cfg.batch_size
        self.stuff_norm_loss = cfg.stuff_norm_loss
        self.instance_triplet_loss = cfg.instance_triplet_loss

        self.local_proposal = cfg.local_proposal

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
        if self.model_mode == 'Centre_clustering' or self.model_mode == 'Zheng_panoptic_wpointnet_PointGroup':
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

            ### centre prediction branch
            ### convolutional occupancy networks decoder
            self.centre_decoder = decoder.LocalDecoder(
                dim=3,
                c_dim=32,
                hidden_size=32,
            )

            module_map = {}

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
                               indice_key_id=1, backbone=True, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            module_map = {}

        ### Our target model without PointNet encoder
        ### Ablation study: figure out the performance improvement of PointNet encoder
        elif self.model_mode == 'Zheng_panoptic_wopointnet_PointGroup':
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

            module_map = {}

        ### only the upper branch of our target model without PointNet encoder
        elif self.model_mode == 'Zheng_upper_wopointnet_PointGroup':
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

            module_map = {
                'module.input_conv': self.input_conv, 'module.unet': self.unet,
                'module.output_layer': self.output_layer,
            }

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

            module_map = {
                'module.input_conv': self.input_conv, 'module.unet': self.unet,
                'module.output_layer': self.output_layer,
                'module.point_refine_feature_attn': self.point_refine_feature_attn,
                'module.atten_outputlayer': self.atten_outputlayer,
            }

        ### same network architecture as PointGroup
        elif self.model_mode == 'Jiang_original_PointGroup':
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, backbone=True, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            #### score branch
            self.score_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1, backbone=False)
            self.score_outputlayer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )
            self.score_linear = nn.Linear(m, 1)

            self.apply(self.set_bn_init)

            module_map = {
                'module.input_conv': self.input_conv, 'module.unet': self.unet,
                'module.output_layer': self.output_layer, 'module.score_unet': self.score_unet,
                'module.score_outputlayer': self.score_outputlayer, 'module.score_linear': self.score_linear
            }

        elif self.model_mode == 'Fan_centre_loss_PointGroup':
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, backbone=True, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.point_deeper_semantic = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, classes, bias=True),
            )

            #### score branch
            self.score_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1)
            self.score_outputlayer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )
            self.score_linear = nn.Linear(m, 1)

            self.apply(self.set_bn_init)

            module_map = {
                'module.input_conv': self.input_conv,
                'module.unet': self.unet,
                'module.output_layer': self.output_layer,
                'module.score_unet': self.score_unet,
                'module.score_outputlayer': self.score_outputlayer,
                'module.score_linear': self.score_linear,
                'module.point_deeper_semantic': self.point_deeper_semantic
            }

        elif self.model_mode == 'Li_simple_backbone_PointGroup':
            self.pointnet_encoder = pointnet.LocalPoolPointnet(
                c_dim=m, dim=6, hidden_dim=m, scatter_type=cfg.scatter_type,
                unet3d=True,
                unet3d_kwargs={"num_levels": cfg.unet3d_num_levels, "f_maps": m, "in_channels": m, "out_channels": m},
                grid_resolution=32, plane_type='grid',
                padding=0.1, n_blocks=5
            )

            self.decoder = decoder.LocalDecoder(
                dim=3,
                c_dim=32,
                hidden_size=32,
            )

            module_map = {'module.pointnet_encoder': self.pointnet_encoder, 'module.decoder': self.decoder}

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

        elif self.model_mode == 'Yu_rc_scorenet_confidence_PointGroup':
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

            self.cluster_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1)
            self.cluster_outputlayer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )
            self.confidence_linear = nn.Linear(m, 1)

            module_map = {
                'module.input_conv': self.input_conv, 'module.unet': self.unet,
                'module.output_layer': self.output_layer,
                'module.point_refine_feature_attn': self.point_refine_feature_attn,
                'module.atten_outputlayer': self.atten_outputlayer,
                'module.cluster_unet': self.cluster_unet, 'module.cluster_outputlayer': self.cluster_outputlayer,
                'module.confidence_linear': self.confidence_linear,
            }

        elif self.model_mode == 'Yu_RC_ScoreNet_Conf_Transformer_PointGroup':
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, backbone=True, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.proposal_transformer = ProposalTransformer(
                d_model=self.m,
                nhead=cfg.Proposal_Transformer['multi_heads'],
                num_decoder_layers=cfg.Proposal_Transformer['num_decoder_layers'],
                dim_feedforward=cfg.Proposal_Transformer['dim_feedforward'],
                dropout=0.0,
            )

            self.proposal_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1)
            self.proposal_outputlayer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )
            self.proposal_confidence_linear = nn.Linear(m, 1)

            module_map = {
                'module.input_conv': self.input_conv,
                'module.unet': self.unet,
                'module.output_layer': self.output_layer,
                'module.proposal_transformer': self.proposal_transformer,
                'module.proposal_unet': self.proposal_unet,
                'module.proposal_outputlayer': self.proposal_outputlayer,
                'module.proposal_confidence_linear': self.proposal_confidence_linear,
            }

        elif self.model_mode == 'Yu_rc_v2_PointGroup':
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

            self.point_feed_forward = nn.Linear(m, m)
            self.point_feed_forward_norm = nn.Sequential(
                norm_fn(m),
                nn.ReLU()
            )

            module_map = {
                'module.input_conv': self.input_conv,
                'module.unet': self.unet,
                'module.output_layer': self.output_layer,
                'module.point_refine_feature_attn': self.point_refine_feature_attn,
                'module.atten_outputlayer': self.atten_outputlayer,
                'module.point_feed_forward': self.point_feed_forward,
                'module.point_feed_forward_norm': self.point_feed_forward_norm
            }

        elif self.model_mode == 'Yu_local_proposal_PointGroup':
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, backbone=True, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.proposal_unet = UBlock([m, 2 * m, 3 * m, 4 * m], norm_fn, 2, block, indice_key_id=1,
                                        backbone=False)
            self.proposal_outputlayer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            module_map = {
                'module.input_conv': self.input_conv,
                'module.unet': self.unet,
                'module.output_layer': self.output_layer,
                'module.proposal_unet': self.proposal_unet,
                'module.proposal_outputlayer': self.proposal_outputlayer,
            }

        elif self.model_mode == 'Yu_stuff_sep_PointGroup':
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

            self.apply(self.set_bn_init)

            module_map = {
                'module.stuff_input_conv': self.stuff_conv,
                'module.stuff_unet': self.stuff_unet,
                'module.stuff_output_layer': self.stuff_output_layer,
                'module.stuff_linear': self.stuff_linear,
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

        elif self.model_mode == 'Centre_sample_cluster':
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

            self.decoder = decoder.LocalDecoder(
                dim=3,
                c_dim=32,
                hidden_size=32,
            )

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

            ### centre prediction branch
            ### convolutional occupancy networks decoder
            self.centre_decoder = decoder.LocalDecoder(
                dim=3,
                c_dim=32,
                hidden_size=32,
            )

            module_map = {}

        elif self.model_mode == 'test_stuff_PointGroup':
            self.stuff_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.stuff_unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, UNet_Transformer=cfg.UNet_Transformer)

            self.stuff_output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.stuff_linear = nn.Linear(m, 2)

            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, backbone=True)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
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

        elif self.model_mode == 'Position_enhanced_PointGroup':
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1, backbone=True, UNet_Transformer=cfg.UNet_Transformer)

            self.output_layer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )

            self.point_position_enhance = nn.Sequential(
                nn.Linear(m+3, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
            )

            #### score branch
            self.score_unet = UBlock([m, 2 * m], norm_fn, 2, block, indice_key_id=1)
            self.score_outputlayer = spconv.SparseSequential(
                norm_fn(m),
                nn.ReLU()
            )
            self.score_linear = nn.Linear(m, 1)

            self.apply(self.set_bn_init)

            module_map = {
                'module.input_conv': self.input_conv,
                'module.unet': self.unet,
                'module.output_layer': self.output_layer,
                'module.point_position_enhance': self.point_position_enhance,
                'module.score_unet': self.score_unet,
                'module.score_outputlayer': self.score_outputlayer,
                'module.score_linear': self.score_linear
            }

        ### point prediction
        self.point_offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU(),
            nn.Linear(m, 3, bias=True),
        )
        module_map['module.point_offset'] = self.point_offset

        if (self.model_mode == 'Yu_stuff_sep_PointGroup') or (self.model_mode == 'Yu_stuff_remove_PointGroup'):
            self.point_semantic = nn.Linear(m, classes - 2)
        elif (self.model_mode == 'Yu_local_proposal_PointGroup'):
            self.point_semantic = nn.Sequential(
                nn.Linear(m, m, bias=True),
                norm_fn(m),
                nn.ReLU(),
                nn.Linear(m, classes, bias=True),
            )
        else:
            self.point_semantic = nn.Linear(m, classes)
        module_map['module.point_semantic'] = self.point_semantic

        #### centre prediction
        ## centre probability
        self.centre_pred = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(),
            nn.Linear(m, 1)
        )
        module_map['module.centre_pre'] = self.centre_pred

        ## centre semantic
        self.centre_semantic = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(),
            nn.Linear(m, classes)
        )
        module_map['module.centre_semantic'] = self.centre_semantic

        ## centre offset
        self.centre_offset = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(),
            nn.Linear(m, 3)
        )
        module_map['module.centre_offset'] = self.centre_offset

        if self.pretrain_path is not None:
            map_location = {'cuda:0': 'cuda:{}'.format(cfg.local_rank)} if cfg.local_rank > 0 else None
            pretrain_dict = torch.load(self.pretrain_path, map_location=map_location)
            if 'module.' in list(pretrain_dict.keys())[0]:
                pretrain_dict = {k[len('module.'):]: v for k, v in pretrain_dict.items()}
            for m in self.pretrain_module:
                n1, n2 = utils.load_model_param(module_map[m], pretrain_dict, prefix=m)
                if cfg.local_rank == 0:
                    print("[PID {}] Load pretrained ".format(os.getpid()) + m + ": {}/{}".format(n1, n2))

        #### fix parameter
        for m in self.fix_module:
            mod = module_map[m]
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
        ret = {}

        batch_idxs = batch_idxs.squeeze()

        if self.model_mode == 'Centre_clustering':
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
                    coords, point_offset_preds[-1], point_semantic_preds, batch_idxs, input['batch_size']
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
                    coords, point_offset_preds[-1], point_semantic_preds, batch_idxs, input['batch_size']
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
                    coords, point_offset_preds[-1], point_semantic_preds, batch_idxs, input['batch_size']
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = semantic_scores
            ret['point_offset_preds'] = point_offset_preds

            ret['centre_preds'] = centre_preds
            ret['centre_semantic_preds'] = centre_semantic_preds
            ret['centre_offset_preds'] = centre_offset_preds

        elif self.model_mode == 'Zheng_upper_wopointnet_PointGroup':
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
            point_semantic_preds = semantic_scores[-1].max(1)[1]

            #### point offset prediction
            point_offset_preds.append(self.point_offset(output_feats))  # (N, 3), float32

            if (epoch == self.test_epoch):
                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_preds[-1], point_semantic_preds, batch_idxs, input['batch_size']
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = semantic_scores
            ret['point_offset_preds'] = point_offset_preds

            if self.stuff_norm_loss:
                ret['output_feats'] = output_feats

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

        elif self.model_mode == 'Jiang_original_PointGroup':
            semantic_scores = []
            point_offset_preds = []

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
            if self.instance_triplet_loss:
                point_offset_pred = point_offset_pred - input['pt_feats'][:, 3:]
            point_offset_preds.append(point_offset_pred)  # (N, 3), float32
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
            if self.instance_triplet_loss:
                ret['point_offset_feats'] = output_feats

        elif self.model_mode == 'Fan_centre_loss_PointGroup':
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

        elif self.model_mode == 'Li_simple_backbone_PointGroup':
            point_offset_preds = []
            point_semantic_scores = []

            point_feats = []
            grid_feats = []

            for sample_indx in range(1, len(batch_offsets)):
                coords_input = coords[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)
                rgb_input = rgb[batch_offsets[sample_indx - 1]:batch_offsets[sample_indx], :].unsqueeze(dim=0)

                encoded_feats = self.pointnet_encoder(coords_input, rgb_input)
                point_feats.append(self.decoder(encoded_feats['coord'], encoded_feats).squeeze(dim=0))
                grid_feats.append(encoded_feats['grid'])

            point_feats = torch.cat(point_feats, 0).contiguous()
            grid_feats = torch.cat(grid_feats, 0).contiguous()

            point_offset_preds.append(self.point_offset(point_feats))

            point_semantic_scores.append(self.point_semantic(point_feats))
            point_semantic_preds = point_semantic_scores[-1].max(1)[1]

            bs, c_dim, grid_size = grid_feats.shape[0], grid_feats.shape[1], grid_feats.shape[2]
            grid_feats = grid_feats.reshape(bs, c_dim, -1).permute(0, 2, 1)

            centre_preds = self.centre_pred(grid_feats)
            centre_semantic_preds = self.centre_semantic(grid_feats)
            centre_offset_preds = self.centre_offset(grid_feats)

            if (epoch == self.test_epoch) and input['test']:
                self.cluster_sets = 'Q'
                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_preds[-1], point_semantic_preds,
                    batch_idxs, input['batch_size']
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = point_semantic_scores
            ret['point_offset_preds'] = point_offset_preds

            ret['centre_preds'] = centre_preds
            ret['centre_semantic_preds'] = centre_semantic_preds
            ret['centre_offset_preds'] = centre_offset_preds

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

        elif self.model_mode == 'Yu_local_proposal_PointGroup':
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

            if (epoch > self.prepare_epochs):
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

                ### local proposal represents that include those proposals around each proposal
                # init all proposal related variables
                local_proposals_idx = []
                local_proposals_offset = []
                local_proposals_offset.append(torch.zeros(1).int())

                proposal_center_coords = []
                for proposal_offset_index in range(1, len(proposals_offset_shift)):
                    points_cluster_index = proposals_idx_shift[
                                     proposals_offset_shift[proposal_offset_index - 1]:
                                     proposals_offset_shift[proposal_offset_index], 1]
                    proposal_center_coords.append(coords[points_cluster_index.long(), :].mean(dim=0, keepdim=True))
                proposal_center_coords = torch.cat(proposal_center_coords, dim=0)

                proposal_dist_mat = euclidean_dist(proposal_center_coords, proposal_center_coords)
                proposal_dist_mat[range(len(proposal_dist_mat)), range(len(proposal_dist_mat))] = 10
                closest_proposals_dist, closest_proposals_index = proposal_dist_mat.topk(k=self.local_proposal['topk'],
                                                                                         dim=1, largest=False)
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
                local_proposals_offset = torch.cat(local_proposals_offset, dim=0)

                #### proposals voxelization again
                input_feats, inp_map = self.clusters_voxelization(
                    local_proposals_idx, local_proposals_offset, output_feats, coords,
                    self.score_fullscale, self.score_scale, self.mode
                )

                #### cluster features
                proposals = self.proposal_unet(input_feats)
                proposals = self.proposal_outputlayer(proposals)
                proposals_point_features = proposals.features[inp_map.long()]  # (sumNPoint, C)

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

            if (epoch == self.test_epoch) and input['test']:
                self.cluster_sets = 'Q'
                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_preds[-1], point_semantic_preds,
                    batch_idxs, input['batch_size']
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = point_semantic_scores
            ret['point_offset_preds'] = point_offset_preds

            if (epoch > self.prepare_epochs):
                ret['output_feats'] = output_feats + refined_point_features
            else:
                ret['output_feats'] = output_feats

        elif self.model_mode == 'Yu_rc_v2_PointGroup':
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

                    c_idxs = proposals_idx_shift[:, 1].cuda()
                    clusters_feats = output_feats[c_idxs.long()]

                    cluster_feature = pointgroup_ops.sec_mean(clusters_feats,
                                                              proposals_offset_shift.cuda())  # (nCluster, m), float

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

                        point_refined_feature = point_refined_feature.squeeze(dim=1) + \
                                                output_feats[batch_offsets[_batch_idx - 1]:batch_offsets[_batch_idx], :]
                        point_refined_feature = self.atten_outputlayer(point_refined_feature)

                        refined_point_features.append(point_refined_feature)

                    refined_point_features = torch.cat(refined_point_features, dim=0)
                    assert refined_point_features.shape[0] == point_features.shape[0], \
                        'point wise features have wrong point numbers'

                    refined_point_features = self.point_feed_forward(refined_point_features) + refined_point_features
                    refined_point_features = self.point_feed_forward_norm(refined_point_features)

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

        elif self.model_mode == 'Yu_rc_scorenet_confidence_PointGroup':
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
                        proposals_idx_shift, proposals_offset_shift, output_feats, coords,
                        self.score_fullscale, self.score_scale, self.mode
                    )

                    #### cluster features
                    clusters = self.cluster_unet(input_feats)
                    clusters = self.cluster_outputlayer(clusters)
                    cluster_feature = clusters.features[inp_map.long()]  # (sumNPoint, C)
                    cluster_feature = pointgroup_ops.roipool(cluster_feature, proposals_offset_shift.cuda())  # (nProposal, C)

                    proposals_confidence_preds.append(self.confidence_linear(cluster_feature))  # (nProposal, 1)
                    proposals_idx_shifts.append(proposals_idx_shift)
                    proposals_offset_shifts.append(proposals_offset_shift)

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

                        point_refined_feature = point_refined_feature.squeeze(dim=1) + \
                                                output_feats[batch_offsets[_batch_idx - 1]:batch_offsets[_batch_idx], :]
                        point_refined_feature = self.atten_outputlayer(point_refined_feature)

                        refined_point_features.append(point_refined_feature)

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

        elif self.model_mode == 'Yu_stuff_sep_PointGroup':
            point_offset_preds = []
            point_semantic_scores = []

            voxel_stuff_feats = pointgroup_ops.voxelization(input['pt_feats'], input['v2p_map'], input['mode'])  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(
                voxel_stuff_feats, input['voxel_coords'],
                input['spatial_shape'], input['batch_size']
            )

            stuff_output = self.stuff_conv(input_)
            stuff_output = self.stuff_unet(stuff_output)
            stuff_output = self.stuff_output_layer(stuff_output)
            stuff_output_feats = stuff_output.features[input_map.long()]
            stuff_output_feats = stuff_output_feats.squeeze(dim=0)

            stuff_preds = self.stuff_linear(stuff_output_feats)

            voxel_feats = pointgroup_ops.voxelization(
                input['pt_feats'], input['v2p_map'], input['mode']
            )  # (M, C), float, cuda

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
            point_semantic_preds = point_semantic_scores[-1].max(1)[1] + 2

            #### point offset prediction
            point_offset_preds.append(self.point_offset(output_feats))  # (N, 3), float32

            if (epoch == self.test_epoch) and input['test']:
                self.cluster_sets = 'Q'

                nonstuff_point_semantic_preds = point_semantic_scores[-1].max(1)[1] + 2
                point_semantic_pred_full = torch.zeros(coords.shape[0], dtype=torch.long).cuda()
                point_semantic_pred_full[
                    (stuff_preds.max(1)[1] == 1).nonzero().squeeze(dim=1).long()] = nonstuff_point_semantic_preds[
                    (stuff_preds.max(1)[1] == 1).nonzero().squeeze(dim=1).long()]

                point_offset_pred = torch.zeros((coords.shape[0], 3), dtype=torch.float).cuda()
                point_offset_pred[(stuff_preds.max(1)[1] == 1).nonzero().squeeze(dim=1).long()] = point_offset_preds[-1][
                    (stuff_preds.max(1)[1] == 1).nonzero().squeeze(dim=1).long()]

                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_pred, point_semantic_pred_full,
                    batch_idxs, input['batch_size'], stuff_preds=stuff_preds.max(1)[1]
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)
                ret['point_semantic_pred_full'] = point_semantic_pred_full

            ret['point_semantic_scores'] = point_semantic_scores
            ret['point_offset_preds'] = point_offset_preds
            ret['output_feats'] = stuff_preds

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

        elif self.model_mode == 'Yu_RC_ScoreNet_Conf_Transformer_PointGroup':
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
                        proposals_idx_shift, proposals_offset_shift, output_feats, coords,
                        self.score_fullscale, self.score_scale, self.mode
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

        elif self.model_mode == 'Centre_sample_cluster':
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
                    centre_queries_coords_input = input['centre_queries_coords'][
                                                  input['centre_queries_batch_offsets'][sample_indx - 1]:
                                                  input['centre_queries_batch_offsets'][sample_indx], :].unsqueeze(dim=0)
                    queries_feats.append(self.decoder(centre_queries_coords_input, decoder_input).squeeze(dim=0))

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

            ### centre prediction
            bs, c_dim, grid_size = grid_feats.shape[0], grid_feats.shape[1], grid_feats.shape[2]
            grid_feats = grid_feats.reshape(bs, c_dim, -1).permute(0, 2, 1)

            centre_preds = self.centre_pred(grid_feats)
            centre_semantic_preds = self.centre_semantic(grid_feats)
            centre_offset_preds = self.centre_offset(grid_feats)

            if not input['test']:
                queries_preds = self.centre_pred(queries_feats)
                queries_semantic_preds = self.centre_semantic(queries_feats)
                queries_offset_preds = self.centre_offset(queries_feats)

            ret['point_semantic_scores'] = semantic_scores
            ret['point_offset_preds'] = point_offset_preds

            ret['centre_preds'] = centre_preds
            ret['centre_semantic_preds'] = centre_semantic_preds
            ret['centre_offset_preds'] = centre_offset_preds

            if not input['test']:
                ret['queries_preds'] = queries_preds
                ret['queries_semantic_preds'] = queries_semantic_preds
                ret['queries_offset_preds'] = queries_offset_preds

        elif self.model_mode == 'test_stuff_PointGroup':
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

            # input_ = spconv.SparseConvTensor(
            #     nonstuff_voxel_feats, nonstuff_voxel_locs, nonstuff_spatial_shape, input['batch_size']
            # )

            input_ = spconv.SparseConvTensor(
                voxel_stuff_feats, input['voxel_coords'],
                input['spatial_shape'], input['batch_size']
            )
            nonstuff_input_map = input_map

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

            if (epoch == self.test_epoch) and ('nonstuff_feats' not in input.keys()):
                self.cluster_sets = 'Q'

                nonstuff_point_semantic_preds = point_semantic_scores[-1].max(1)[1]

                stuff_valid = torch.zeros(coords.shape[0], dtype=torch.long).cuda()
                stuff_valid[(stuff_preds.max(1)[1] == 1).nonzero().squeeze(dim=1).long()] = 1
                stuff_valid[(stuff_preds.softmax(dim=1).max(1)[0] < 0.0).nonzero().squeeze(dim=1).long()] = 0
                # stuff_valid[(nonstuff_point_semantic_preds.max(1)[1] > 1).nonzero().squeeze(dim=1).long()] = 1

                point_semantic_pred_full = torch.zeros(coords.shape[0], dtype=torch.long).cuda()
                point_semantic_pred_full[stuff_valid.nonzero()] = nonstuff_point_semantic_preds[stuff_valid.nonzero()]

                point_offset_pred = torch.zeros((coords.shape[0], 3), dtype=torch.float).cuda()
                point_offset_pred[stuff_valid.nonzero()] = nonstuff_point_offset_pred[stuff_valid.nonzero()]

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

        elif self.model_mode == 'Position_enhanced_PointGroup':
            semantic_scores = []
            point_offset_preds = []

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

            point_semantic_preds = semantic_scores[0].max(1)[1]

            point_position_encoding = self.point_position_enhance(
                torch.cat((output_feats, input['pt_feats'][:, 3:]), dim=-1))

            #### point offset prediction
            point_offset_feats = output_feats + point_position_encoding
            point_offset_pred = self.point_offset(point_offset_feats)  # (N, 3), float32
            if self.instance_triplet_loss:
                point_offset_pred = point_offset_pred - input['pt_feats'][:, 3:]
            point_offset_preds.append(point_offset_pred)
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
            if self.instance_triplet_loss:
                ret['point_offset_feats'] = point_offset_feats

        return ret
