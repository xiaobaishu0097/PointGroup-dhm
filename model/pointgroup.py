import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import spconv
import functools
import sys

sys.path.append('../')

from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
from model.encoder import pointnet, pointnetpp
from model.encoder.unet3d import UNet3D
from model.decoder import decoder
from model.common import coordinate2index, normalize_3d_coordinate

from model.components import ResidualBlock, VGGBlock, UBlock, backbone_pointnet2


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
        self.refine_times = cfg.refine_times
        self.add_pos_enc_ref = cfg.add_pos_enc_ref

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
                               indice_key_id=1)

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

            module_map = {}

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

            module_map = {
                'module.input_conv': self.input_conv, 'module.unet': self.unet,
                'module.output_layer': self.output_layer,
            }

        elif self.model_mode == 'Yu_refine_clustering_PointGroup':
            self.input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
            )

            self.unet = UBlock([m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m], norm_fn, block_reps, block,
                               indice_key_id=1)

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

            module_map = {
                'module.input_conv': self.input_conv, 'module.unet': self.unet,
                'module.output_layer': self.output_layer, 'module.score_unet': self.score_unet,
                'module.score_outputlayer': self.score_outputlayer, 'module.score_linear': self.score_linear
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
                               indice_key_id=1)

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

        elif self.model_mode == 'PointNet_point_prediction_test_PointGroup':
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

            module_map = {
                'module.pointnet_encoder': self.pointnet_encoder
            }

        ### point prediction
        self.point_offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU(),
            nn.Linear(m, 3, bias=True),
        )
        module_map['module.point_offset'] = self.point_offset

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
            pretrain_dict = torch.load(self.pretrain_path)
            for m in self.pretrain_module:
                print(
                    "Load pretrained " + m + ": %d/%d" % utils.load_model_param(module_map[m], pretrain_dict, prefix=m)
                )

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

    def forward(self, input, input_map, coords, rgb, ori_coords, point_positional_encoding, batch_idxs, batch_offsets, epoch):
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

                for _ in range(self.refine_times):
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
                    if self.add_pos_enc_ref:
                        refined_point_features += point_positional_encoding
                    point_features = refined_point_features.clone()

                    ### refined point prediction
                    #### refined point semantic label prediction
                    point_semantic_scores.append(self.point_semantic(refined_point_features))  # (N, nClass), float
                    point_semantic_preds = point_semantic_scores[-1].max(1)[1]

                    #### point offset prediction
                    point_offset_preds.append(self.point_offset(refined_point_features))  # (N, 3), float32

                if (epoch == self.test_epoch):
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
            # only used to evaluate based on ground truth
            # semantic_scores.append(input['point_semantic_scores'][0])  # (N, nClass), float
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

            if (epoch == self.test_epoch):
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

                for _ in range(self.refine_times):
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
                    if self.add_pos_enc_ref:
                        refined_point_features += point_positional_encoding
                    point_features = refined_point_features.clone()

                    ### refined point prediction
                    #### refined point semantic label prediction
                    point_semantic_scores.append(self.point_semantic(refined_point_features))  # (N, nClass), float
                    point_semantic_preds = point_semantic_scores[-1].max(1)[1]

                    #### point offset prediction
                    point_offset_preds.append(self.point_offset(refined_point_features))  # (N, 3), float32

                if (epoch == self.test_epoch):
                    self.cluster_sets = 'Q'
                    scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                        coords, point_offset_preds[-1], point_semantic_preds,
                        batch_idxs, input['batch_size']
                    )
                    ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = point_semantic_scores
            ret['point_offset_preds'] = point_offset_preds

        elif self.model_mode == 'PointNet_point_prediction_test_PointGroup':
            point_offset_preds = []
            point_semantic_scores = []

            point_feats, _ = self.pointnet_backbone_forward(coords, ori_coords, rgb, batch_offsets)

            ### point prediction
            #### point semantic label prediction
            point_semantic_scores.append(self.point_semantic(point_feats))  # (N, nClass), float
            point_semantic_preds = point_semantic_scores[-1].max(1)[1]

            #### point offset prediction
            point_offset_preds.append(self.point_offset(point_feats))  # (N, 3), float32

            if (epoch == self.test_epoch):
                scores, proposals_idx, proposals_offset = self.pointgroup_cluster_algorithm(
                    coords, point_offset_preds[-1], point_semantic_preds, batch_idxs, input['batch_size']
                )
                ret['proposal_scores'] = (scores, proposals_idx, proposals_offset)

            ret['point_semantic_scores'] = point_semantic_scores
            ret['point_offset_preds'] = point_offset_preds

        return ret
