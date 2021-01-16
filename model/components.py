import torch
import torch.nn as nn
import spconv
from spconv.modules import SparseModule
from collections import OrderedDict
import sys

sys.path.append('../')

# from model.Pointnet2.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModule


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


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=0.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        at = self.alpha.gather(0, targets.data.view(-1).to(torch.long))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt.view(-1)) ** self.gamma * BCE_loss.view(-1)
        return F_loss.mean()
