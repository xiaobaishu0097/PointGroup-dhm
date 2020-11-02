# import multiprocessing
import torch
import numpy as np
import math
import numbers
from math import exp

from torch import nn
from torch.nn import functional as F


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def chamfer_distance(points1, points2, use_kdtree=True, give_id=False):
    ''' Returns the chamfer distance for the sets of points.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        use_kdtree (bool): whether to use a kdtree
        give_id (bool): whether to return the IDs of nearest points
    '''

    return chamfer_distance_naive(points1, points2)


def chamfer_distance_naive(points1, points2):
    ''' Naive implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
    '''
    assert (points1.size() == points2.size())
    batch_size, T, _ = points1.size()

    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)

    distances = (points1 - points2).pow(2).sum(-1)

    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)

    chamfer = chamfer1 + chamfer2
    return chamfer


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def transform_points(points, transform):
    ''' Transforms points with regard to passed camera information.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    assert (points.size(2) == 3)
    assert (transform.size(1) == 3)
    assert (points.size(0) == transform.size(0))

    if transform.size(2) == 4:
        R = transform[:, :, :3]
        t = transform[:, :, 3:]
        points_out = points @ R.transpose(1, 2) + t.transpose(1, 2)
    elif transform.size(2) == 3:
        K = transform
        points_out = points @ K.transpose(1, 2)

    return points_out


def b_inv(b_mat):
    ''' Performs batch matrix inversion.

    Arguments:
        b_mat: the batch of matrices that should be inverted
    '''

    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv


def project_to_camera(points, transform):
    ''' Projects points to the camera plane.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    p_camera = transform_points(points, transform)
    p_camera = p_camera[..., :2] / p_camera[..., 2:]
    return p_camera


def fix_Rt_camera(Rt, loc, scale):
    ''' Fixes Rt camera matrix.

    Args:
        Rt (tensor): Rt camera matrix
        loc (tensor): location
        scale (float): scale
    '''
    # Rt is B x 3 x 4
    # loc is B x 3 and scale is B
    batch_size = Rt.size(0)
    R = Rt[:, :, :3]
    t = Rt[:, :, 3:]

    scale = scale.view(batch_size, 1, 1)
    R_new = R * scale
    t_new = t + R @ loc.unsqueeze(2)

    Rt_new = torch.cat([R_new, t_new], dim=2)

    assert (Rt_new.size() == (batch_size, 3, 4))
    return Rt_new


def normalize_coordinate(p, padding=0.1, plane='xz'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane == 'xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
    xy_new = xy_new + 0.5  # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def normalize_3d_coordinate(p, padding=0.1):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    p_nor = p - p.min(dim=1, keepdim=True)[0]
    p_nor = p_nor / p_nor.max(dim=1, keepdim=True)[0]
    # p_nor -= 0.5
    # p_nor = p_nor / (1 + padding + 10e-4)  # (-0.5, 0.5)
    # p_nor = p_nor + 0.5  # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def normalize_coord(p, vol_range, plane='xz'):
    ''' Normalize coordinate to [0, 1] for sliding-window experiments

    Args:
        p (tensor): point
        vol_range (numpy array): volume boundary
        plane (str): feature type, ['xz', 'xy', 'yz'] - canonical planes; ['grid'] - grid volume
    '''
    p[:, 0] = (p[:, 0] - vol_range[0][0]) / (vol_range[1][0] - vol_range[0][0])
    p[:, 1] = (p[:, 1] - vol_range[0][1]) / (vol_range[1][1] - vol_range[0][1])
    p[:, 2] = (p[:, 2] - vol_range[0][2]) / (vol_range[1][2] - vol_range[0][2])

    if plane == 'xz':
        x = p[:, [0, 2]]
    elif plane == 'xy':
        x = p[:, [0, 1]]
    elif plane == 'yz':
        x = p[:, [1, 2]]
    else:
        x = p
    return x


def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d':  # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d':  # grid
        index = x[:, :, 2] + reso * (x[:, :, 1] + reso * x[:, :, 0])
    index = index[:, None, :]
    return index


def coord2index(p, vol_range, reso=None, plane='xz'):
    ''' Normalize coordinate to [0, 1] for sliding-window experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): points
        vol_range (numpy array): volume boundary
        reso (int): defined resolution
        plane (str): feature type, ['xz', 'xy', 'yz'] - canonical planes; ['grid'] - grid volume
    '''
    # normalize to [0, 1]
    x = normalize_coord(p, vol_range, plane=plane)

    if isinstance(x, np.ndarray):
        x = np.floor(x * reso).astype(int)
    else:  # * pytorch tensor
        x = (x * reso).long()

    if x.shape[1] == 2:
        index = x[:, 0] + reso * x[:, 1]
        index[index > reso ** 2] = reso ** 2
    elif x.shape[1] == 3:
        index = x[:, 0] + reso * (x[:, 1] + reso * x[:, 2])
        index[index > reso ** 3] = reso ** 3

    return index[None]


def update_reso(reso, depth):
    ''' Update the defined resolution so that UNet can process.

    Args:
        reso (int): defined resolution
        depth (int): U-Net number of layers
    '''
    base = 2 ** (int(depth) - 1)
    if ~(reso / base).is_integer():  # when this is not integer, U-Net dimension error
        for i in range(base):
            if ((reso + i) / base).is_integer():
                reso = reso + i
                break
    return reso


def decide_total_volume_range(query_vol_metric, recep_field, unit_size, unet_depth):
    ''' Update the defined resolution so that UNet can process.

    Args:
        query_vol_metric (numpy array): query volume size
        recep_field (int): defined the receptive field for U-Net
        unit_size (float): the defined voxel size
        unet_depth (int): U-Net number of layers
    '''
    reso = query_vol_metric / unit_size + recep_field - 1
    reso = update_reso(int(reso), unet_depth)  # make sure input reso can be processed by UNet
    input_vol_metric = reso * unit_size
    p_c = np.array([0.0, 0.0, 0.0]).astype(np.float32)
    lb_input_vol, ub_input_vol = p_c - input_vol_metric / 2, p_c + input_vol_metric / 2
    lb_query_vol, ub_query_vol = p_c - query_vol_metric / 2, p_c + query_vol_metric / 2
    input_vol = [lb_input_vol, ub_input_vol]
    query_vol = [lb_query_vol, ub_query_vol]

    # handle the case when resolution is too large
    if reso > 10000:
        reso = 1

    return input_vol, query_vol, reso


def add_key(base, new, base_name, new_name, device=None):
    ''' Add new keys to the given input

    Args:
        base (tensor): inputs
        new (tensor): new info for the inputs
        base_name (str): name for the input
        new_name (str): name for the new info
        device (device): pytorch device
    '''
    if (new is not None) and (isinstance(new, dict)):
        if device is not None:
            for key in new.keys():
                new[key] = new[key].to(device)
        base = {base_name: base,
                new_name: new}
    return base


class map2local(object):
    ''' Add new keys to the given input

    Args:
        s (float): the defined voxel size
        pos_encoding (str): method for the positional encoding, linear|sin_cos
    '''

    def __init__(self, s, pos_encoding='linear'):
        super().__init__()
        self.s = s
        self.pe = positional_encoding(basis_function=pos_encoding)

    def __call__(self, p):
        p = torch.remainder(p, self.s) / self.s  # always possitive
        # p = torch.fmod(p, self.s) / self.s # same sign as input p!
        p = self.pe(p)
        return p


class positional_encoding(object):
    ''' Positional Encoding (presented in NeRF)

    Args:
        basis_function (str): basis function
    '''

    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 10
        freq_bands = 2. ** (np.linspace(0, L - 1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = 2.0 * p - 1.0  # chagne to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def generate_heatmap(grid_xyz, instance_centers, sigma=0.25):
    scaledGaussian = lambda x: exp(-(1 / 2) * ((x / sigma) ** 2))

    distance = torch.tensor(grid_xyz).unsqueeze(dim=1).repeat(1, len(instance_centers), 1) - torch.tensor(
        instance_centers).unsqueeze(dim=0).repeat(
        grid_xyz.shape[0], 1, 1)
    heatmap = torch.norm(distance, dim=2)
    heatmap = heatmap.apply_(scaledGaussian)
    heatmap = heatmap.max(dim=1)[0]
    heatmap[heatmap < exp(-1)] = 0

    return heatmap


def generate_adaptive_heatmap(
        grid_xyz: torch.tensor,  # (N, 3)
        instance_centers: torch.tensor,  # (N, 3)
        instance_size: torch.tensor,  # (N, 3)
        instance_label: torch.tensor = None, # (N, 1)
        min_IoU=0.5,
        min_radius=0,
):
    size_adaptive_radius = lambda x: (1 - np.sqrt((2 * min_IoU) / (1 + min_IoU))) * x
    # we assume that the bounding box is the same as ground truth
    # then r = (1 - \frac{2 IoU}{1 + IoU}) \sqrt{w^2 + h^2 + d^2}
    scaledGaussian = lambda x: exp(-(1 / 2) * (x ** 2))

    diagonal_length = torch.norm(instance_size, dim=1, keepdim=True)
    instance_radius = diagonal_length.apply_(size_adaptive_radius)
    instance_radius[instance_radius < min_radius] = torch.tensor(min_radius)
    instance_sigma = instance_radius / 3

    distance = grid_xyz.unsqueeze(dim=1).repeat(1, instance_centers.shape[0], 1) - instance_centers.unsqueeze(
        dim=0).repeat(grid_xyz.shape[0], 1, 1)
    grid_instance_distance = torch.norm(distance, dim=2)

    heatmap = torch.div(grid_instance_distance,
                        instance_sigma.reshape(1, -1).repeat(grid_instance_distance.shape[0], 1))
    heatmap = heatmap.apply_(scaledGaussian)
    heatmap, instance_label_indexs = heatmap.max(dim=1)
    heatmap[heatmap < exp(-1)] = 0

    if instance_label is not None:
        grid_instance_label = instance_label[instance_label_indexs]

        invalid_grid_index = grid_instance_label == 20
        heatmap[invalid_grid_index] = 0

        return {
            'heatmap': heatmap,
            'grid_instance_label': grid_instance_label,
        }

    return {
        'heatmap': heatmap,
    }
