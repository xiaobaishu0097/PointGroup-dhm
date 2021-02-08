'''
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by Li Jiang
'''

import os, sys, glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import Dataset, DataLoader
import torch_scatter

from model.common import generate_heatmap, normalize_3d_coordinate, coordinate2index, generate_adaptive_heatmap

sys.path.append('../')

from util.config import cfg
from util.log import logger
from lib.pointgroup_ops.functions import pointgroup_ops

SEMANTIC_NAME2INDEX = {
    'wall': 0, 'floor': 1, 'cabinet': 2, 'bed': 3, 'chair': 4, 'sofa': 5, 'table': 6, 'door': 7, 'window': 8,
    'bookshelf': 9, 'picture': 10, 'counter': 11, 'desk': 12, 'curtain': 13, 'refridgerator': 14,
    'shower curtain': 15, 'toilet': 16, 'sink': 17, 'bathtub': 18, 'otherfurniture': 19
}

class ScannetDatast:
    def __init__(self, test=False):
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.filename_suffix = cfg.filename_suffix

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode

        self.heatmap_sigma = cfg.heatmap_sigma
        self.min_IoU = cfg.min_IoU

        self.remove_class = []
        for class_name in cfg.remove_class:
            self.remove_class.append(SEMANTIC_NAME2INDEX[class_name])

        if test:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1

    def trainLoader(self):
        if not cfg.overfitting:
            file_names = sorted(
                glob.glob(os.path.join(self.data_root, self.dataset, 'train', '*' + self.filename_suffix))
            )
            self.train_data_files = [torch.load(i) for i in file_names]
        else:
            file_names = sorted(
                glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*' + self.filename_suffix))
            )
            self.train_data_files = [torch.load(file_names[0])]

        logger.info('Training samples: {}'.format(len(self.train_data_files)))
        self.train_set = list(range(len(self.train_data_files)))

    def valLoader(self):
        file_names = sorted(
            glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*' + self.filename_suffix))
        )
        self.val_data_files = [torch.load(i) for i in file_names]

        logger.info('Validation samples: {}'.format(len(self.val_data_files)))
        self.val_set = list(range(len(self.val_data_files)))


    def testLoader(self):
        if not cfg.overfitting:
            self.test_file_names = sorted(
                glob.glob(os.path.join(self.data_root, self.dataset, self.test_split, '*' + self.filename_suffix))
            )
            self.test_data_files = [torch.load(i) for i in self.test_file_names]
        else:
            self.test_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*' + self.filename_suffix)))
            self.test_data_files = [torch.load(self.test_file_names[0])]

        logger.info('Testing samples ({}): {}'.format(self.test_split, len(self.test_data_files)))
        self.test_set = list(range(len(self.test_data_files)))

    #Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x + g(x) * mag


    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # (nInst), int
        instance_num = int(instance_label.max()) + 1
        instance_centres = []
        instance_sizes = []
        instance_labels = []
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            size_xyz_i = xyz_i.max(0) - xyz_i.min(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ret = {}

            ### instance_centres
            instance_centres.append(mean_xyz_i)
            instance_sizes.append(size_xyz_i)
            if semantic_label is not None:
                semantic_label[semantic_label == -100] = 20
                instance_labels.append(np.argmax(np.bincount(np.int32(semantic_label[inst_idx_i]))))
                ret['instance_label'] = instance_labels

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

            ret['instance_info'] = instance_info
            ret['instance_pointnum'] = instance_pointnum
            ret['instance_centre'] = instance_centres
            ret['instance_size'] = instance_sizes

        return instance_num, ret


    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)


    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs


    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    def padding_pointcloud_data(self, xyz, rgb, label, instance_label):
        point_num = xyz.shape[0]

        xyz_padding = np.zeros((self.max_point_num - point_num, 3), dtype=np.float32)
        xyz_padded = np.concatenate((xyz, xyz_padding), axis=0)

        rgb_padding = np.ones((self.max_point_num - point_num, 3), dtype=np.float32)
        rgb_padded = np.concatenate((rgb, rgb_padding), axis=0)

        label_padding = np.ones((self.max_point_num - point_num, )) * -100
        label_padded = np.concatenate((label, label_padding), axis=0)

        instance_label_padding = np.ones((self.max_point_num - point_num, )) * -100
        instance_label_padded = np.concatenate((instance_label, instance_label_padding), axis=0)

        return xyz_padded, rgb_padded, label_padded, instance_label_padded

    def positional_encoding_func(self, position, embed_dim):
        pos_en = []
        for pos_idx in range(position.shape[1]):
            pos = position[:, pos_idx]
            for d in range(embed_dim):
                pos_en.append(np.sin(np.array(np.pi) * (2 ** d) * pos).unsqueeze(dim=1))
                pos_en.append(np.cos(np.array(np.pi) * (2 ** d) * pos).unsqueeze(dim=1))
        pos_en = torch.cat(pos_en, dim=1)

        return pos_en

    def trainMerge(self, id):
        ret_dict = {}

        # variables for backbone
        point_locs = [] # (N, 4) (sample_index, xyz)
        point_coords = []  # (N, 6) (shifted_xyz, original_xyz)
        point_feats = []  # (N, 3) (rgb)

        # variables for point-wise predictions
        point_semantic_labels = []  # (N)
        point_instance_labels = []  # (N)
        point_instance_infos = []  # (N, 9)
        instance_centres = []  # (nInst, 3) (instance_xyz)

        # variables for grid-wise predictions
        grid_centre_heatmaps = []  # (B, nGrid)
        grid_centre_indicators = []  # (nInst) --> (B, nGrid)
        grid_centre_offset_labels = []  # (nInst, 3) --> (B, nGrid, 3)
        grid_centre_semantic_labels = []  # (B, nGrid)

        # variables for centre queries
        centre_queries_coords = []
        centre_queries_probs = []
        centre_queries_semantic_labels = []
        centre_queries_offsets = []
        centre_queries_batch_offsets = [0]

        #
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]
        total_inst_num = 0

        for i, idx in enumerate(id):
            xyz_origin, rgb, label, instance_label = self.train_data_files[idx]

            for class_idx in self.remove_class:
                valid_class_indx = (label != class_idx)
                xyz_origin = xyz_origin[valid_class_indx, :]
                rgb = rgb[valid_class_indx, :]
                label = label[valid_class_indx]
                instance_label = instance_label[valid_class_indx]

            ### jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, True, True, True)
            # xyz_middle = self.dataAugment(xyz_origin, False, False, False)

            ### scale
            xyz = xyz_middle * self.scale

            ### elastic
            xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

            ### offset
            xyz -= xyz.min(0)

            ### crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_origin = xyz_origin[valid_idxs]
            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            # label[label == -100] = 20

            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), label.copy())
            instance_infos = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list
            inst_centres = inst_infos['instance_centre']  # (nInst, 3) (cx, cy, cz)
            inst_size = inst_infos['instance_size']
            inst_label = inst_infos['instance_label']

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            if 'Centre' in cfg.model_mode.split('_'):
                ### get instance centre heatmap
                grid_xyz = np.zeros((32 ** 3, 3), dtype=np.float32)
                grid_xyz += xyz_middle.min(axis=0, keepdims=True)
                grid_size = (xyz_middle.max(axis=0, keepdims=True) - xyz_middle.min(axis=0, keepdims=True)) / 32
                grid_xyz += grid_size / 2
                grid_xyz = grid_xyz.reshape(32, 32, 32, 3)
                for index in range(32):
                    grid_xyz[index, :, :, 0] = grid_xyz[index, :, :, 0] + index * grid_size[0, 0]
                    grid_xyz[:, index, :, 1] = grid_xyz[:, index, :, 1] + index * grid_size[0, 1]
                    grid_xyz[:, :, index, 2] = grid_xyz[:, :, index, 2] + index * grid_size[0, 2]
                grid_xyz = grid_xyz.reshape(-1, 3)

                ### size adaptive gaussian function or fixed sigma gaussian
                if not self.heatmap_sigma:
                    grid_infos = generate_adaptive_heatmap(
                        torch.tensor(grid_xyz, dtype=torch.float64), torch.tensor(inst_centres), torch.tensor(inst_size),
                        torch.tensor(inst_label), min_IoU=self.min_IoU, min_radius=np.linalg.norm(grid_size),
                    )
                    grid_centre_heatmap = grid_infos['heatmap']
                    grid_instance_label = grid_infos['grid_instance_label']
                    grid_instance_label[grid_instance_label == 20] = -100
                else:
                    grid_centre_heatmap = generate_heatmap(grid_xyz.astype(np.double), np.asarray(inst_centres),
                                                    sigma=self.heatmap_sigma)

                ### sample centre queries around instance centres
                centre_queries_coord = []
                centre_queries_offset = []
                n_queries = 100
                for inst_cent in inst_centres:
                    centre_queries_offset.append(torch.randn(n_queries, 3).double() * 0.05)
                    centre_query_coords = torch.from_numpy(inst_cent) - centre_queries_offset[-1]
                    centre_queries_coord.append(centre_query_coords)

                centre_queries_coord = torch.cat(centre_queries_coord, dim=0)
                centre_queries_offset = torch.cat(centre_queries_offset, dim=0)

                centre_queries_infos = generate_adaptive_heatmap(
                    centre_queries_coord, torch.tensor(inst_centres), torch.tensor(inst_size),
                    torch.tensor(inst_label), min_IoU=self.min_IoU, min_radius=np.linalg.norm(grid_size),
                )
                centre_queries_prob = centre_queries_infos['heatmap']
                centre_queries_semantic_label = centre_queries_infos['grid_instance_label']
                centre_queries_semantic_label[centre_queries_semantic_label == 20] = -100
                centre_queries_offset[centre_queries_semantic_label == -100] = torch.zeros_like(centre_queries_offset[0, :])

                centre_queries_batch_offsets.append(centre_queries_batch_offsets[-1] + centre_queries_coord.shape[0])

                norm_coords = normalize_3d_coordinate(
                    torch.cat((torch.from_numpy(xyz_middle), torch.from_numpy(np.asarray(inst_centres))), dim=0).unsqueeze(
                        dim=0).clone()
                )
                norm_inst_centres = norm_coords[:, -len(inst_centres):, :]
                grid_centre_gt = coordinate2index(norm_inst_centres, 32, coord_type='3d').squeeze()
                if grid_centre_gt.ndimension() == 0:
                    grid_centre_gt = grid_centre_gt.unsqueeze(dim=0)
                assert grid_centre_gt.ndimension() == 1, 'the dimension of grid_centre_gt is {}'.format(
                    grid_centre_gt.ndimension())
                grid_centre_indicator = torch.cat(
                    (torch.LongTensor(grid_centre_gt.shape[-1], 1).fill_(i), grid_centre_gt.unsqueeze(dim=-1)), dim=1
                )

                ### only the centre grid point require to predict the offset vector
                grid_cent_xyz = grid_xyz[grid_centre_gt]
                grid_centre_offset = grid_cent_xyz - np.array(inst_centres)
                grid_centre_offset = grid_centre_offset.squeeze()
                grid_centre_offset = torch.from_numpy(grid_centre_offset)
                if grid_centre_offset.ndimension() == 1:
                    grid_centre_offset = grid_centre_offset.unsqueeze(dim=0)
                assert grid_centre_offset.ndimension() == 2, 'the dimension of grid_centre_gt is {}'.format(
                    grid_centre_offset.ndimension())
                grid_centre_offset_label = torch.cat(
                    (torch.DoubleTensor(grid_centre_offset.shape[0], 1).fill_(i), grid_centre_offset), dim=1
                )

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            # variables for backbone
            point_locs.append(torch.cat((torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()), dim=1))  # (N, 4) (sample_index, xyz)
            point_coords.append(torch.from_numpy(np.concatenate((xyz_middle, xyz_origin), axis=1)))  # (N, 6) (shifted_xyz, original_xyz)
            point_feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)  # (N, 3) (rgb)
            # variables for point-wise predictions
            point_semantic_labels.append(torch.from_numpy(label))  # (N)
            point_instance_labels.append(torch.from_numpy(instance_label))  # (N)
            point_instance_infos.append(torch.from_numpy(instance_infos))  # (N, 9) (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            instance_centres.append(
                torch.cat(
                    (torch.DoubleTensor(len(inst_centres), 1).fill_(i), torch.from_numpy(np.array(inst_centres))),
                    dim=1
                )
            )  # (nInst, 4) (sample_index, instance_centre_xyz)

            if 'Centre' in cfg.model_mode.split('_'):
                # variables for grid-wise predictions
                grid_centre_heatmaps.append(grid_centre_heatmap.unsqueeze(dim=0))  # (B, nGrid)
                grid_centre_indicators.append(grid_centre_indicator)  # (NGrid, 2) (sample_index, grid_index)
                grid_centre_offset_labels.append(grid_centre_offset_label)  # (NGrid, 4) (sample_index, grid_centre_offset)
                grid_centre_semantic_labels.append(grid_instance_label.unsqueeze(dim=0))  # (B, nGrid)

                centre_queries_coords.append(centre_queries_coord)
                centre_queries_probs.append(centre_queries_prob)
                centre_queries_semantic_labels.append(centre_queries_semantic_label)
                centre_queries_offsets.append(centre_queries_offset)

            # variable for other uses
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)

        point_locs = torch.cat(point_locs, 0)  # (N) (sample_index)
        point_coords = torch.cat(point_coords, 0).to(torch.float32)  # (N, 6) (shifted_xyz, original_xyz)
        point_feats = torch.cat(point_feats, 0)  # (N, 3) (rgb)
        if cfg.pos_enc == 'XYZ':
            point_positional_encoding = torch.cat(
                (
                    torch.zeros(point_coords.shape[0], 1), self.positional_encoding_func(point_coords[:, :3], 5),
                    torch.zeros(point_coords.shape[0], 1)
                ), dim=1
            )
        elif cfg.pos_enc == 'XYZRGB':
            point_positional_encoding = torch.cat(
                (
                    torch.zeros(point_coords.shape[0], 1), self.positional_encoding_func(point_coords[:, :3], 3),
                    torch.zeros(point_coords.shape[0], 1), self.positional_encoding_func(point_feats, 2)
                ), dim=1
            )
        # variables for point-wise predictions
        point_semantic_labels = torch.cat(point_semantic_labels, 0).long()  # (N)
        point_instance_labels = torch.cat(point_instance_labels, 0).long()  # (N)
        point_instance_infos = torch.cat(point_instance_infos, 0).to(torch.float32) # (N, 9) (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_centres = torch.cat(instance_centres, 0)  # (nInst, 3) (instance_centre_xyz)

        # variables for backbone
        ret_dict['point_locs'] = point_locs # (N, 4) (sample_index, xyz)
        ret_dict['point_coords'] = point_coords # (N, 6) (shifted_xyz, original_xyz)
        ret_dict['point_feats'] = point_feats # (N, 3) (rgb)
        ret_dict['point_positional_encoding'] = point_positional_encoding # (N, 32) (0, xyz-18dim, 0, rgb-12dim)
        ret_dict['point_semantic_labels'] = point_semantic_labels  # (N)
        ret_dict['point_instance_labels'] = point_instance_labels  # (N)
        ret_dict['point_instance_infos'] = point_instance_infos  # (N, 9)
        ret_dict['instance_centres'] = instance_centres  # (nInst, 3) (instance_xyz)

        if 'Centre' in cfg.model_mode.split('_'):
            # variables for grid-wise predictions
            grid_centre_heatmaps = torch.cat(grid_centre_heatmaps, 0).to(torch.float32)  # (B, nGrid)
            grid_centre_indicators = torch.cat(grid_centre_indicators, 0)  # (NInst, 2) (sample_index, grid_index)
            grid_centre_offset_labels = torch.cat(grid_centre_offset_labels, 0).to(torch.float32)  # (NInst, 4) (sample_index, grid_centre_offset)
            grid_centre_semantic_labels = torch.cat(grid_centre_semantic_labels, 0)  # (B, nGrid)

            centre_queries_coords = torch.cat(centre_queries_coords, dim=0)
            centre_queries_probs = torch.cat(centre_queries_probs, dim=0).to(torch.float32)
            centre_queries_semantic_labels = torch.cat(centre_queries_semantic_labels, dim=0)
            centre_queries_offsets = torch.cat(centre_queries_offsets, dim=0).to(torch.float32)
            centre_queries_batch_offsets = torch.tensor(centre_queries_batch_offsets, dtype=torch.int)

            # variables for grid-wise predictions
            ret_dict['grid_centre_heatmap'] = grid_centre_heatmaps  # (B, nGrid)
            ret_dict['grid_centre_indicator'] = grid_centre_indicators  # (NGrid, 2) (sample_index, grid_index)
            ret_dict['grid_centre_offset_labels'] = grid_centre_offset_labels  # (NGrid, 4) (sample_index, grid_centre_offset)
            ret_dict['grid_centre_semantic_labels'] = grid_centre_semantic_labels  # (B, nGrid)

            ret_dict['centre_queries_coords'] = centre_queries_coords
            ret_dict['centre_queries_probs'] = centre_queries_probs
            ret_dict['centre_queries_semantic_labels'] = centre_queries_semantic_labels
            ret_dict['centre_queries_offsets'] = centre_queries_offsets
            ret_dict['centre_queries_batch_offsets'] = centre_queries_batch_offsets

        # variable for other uses
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int) # int (total_nInst)

        spatial_shape = np.clip((point_locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(point_locs, self.batch_size, self.mode)

        # variables for point-wise predictions
        ret_dict['voxel_locs'] = voxel_locs  # (nVoxel, 4)
        ret_dict['p2v_map'] = p2v_map  # (N)
        ret_dict['v2p_map'] = v2p_map  # (nVoxel, 19?)
        # variables for other uses
        ret_dict['instance_pointnum'] = instance_pointnum  # (nInst) # currently used in Jiang_PointGroup
        ret_dict['id'] = id
        ret_dict['batch_offsets'] = batch_offsets  # int (B+1)
        ret_dict['spatial_shape'] = spatial_shape  # long (3)

        if 'stuff' in cfg.model_mode.split('_'):
            nonstuff_spatial_shape = np.clip(
                (point_locs[point_semantic_labels > 1].max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)
            nonstuff_voxel_locs, nonstuff_p2v_map, nonstuff_v2p_map = pointgroup_ops.voxelization_idx(
                point_locs[point_semantic_labels > 1], self.batch_size, self.mode
            )

            # nonstuff related
            ret_dict['nonstuff_spatial_shape'] = nonstuff_spatial_shape
            ret_dict['nonstuff_voxel_locs'] = nonstuff_voxel_locs
            ret_dict['nonstuff_p2v_map'] = nonstuff_p2v_map
            ret_dict['nonstuff_v2p_map'] = nonstuff_v2p_map

        return ret_dict

    def valMerge(self, id):
        ret_dict = {}

        # variables for backbone
        point_locs = [] # (N, 4) (sample_index, xyz)
        point_coords = []  # (N, 6) (shifted_xyz, original_xyz)
        point_feats = []  # (N, 3) (rgb)

        # variables for point-wise predictions
        point_semantic_labels = []  # (N)
        point_instance_labels = []  # (N)
        point_instance_infos = []  # (N, 9)
        instance_centres = []  # (nInst, 3) (instance_xyz)

        # variables for grid-wise predictions
        grid_centre_heatmaps = []  # (B, nGrid)
        grid_centre_indicators = []  # (nInst) --> (B, nGrid)
        grid_centre_offset_labels = []  # (nInst, 3) --> (B, nGrid, 3)
        grid_centre_semantic_labels = []  # (B, nGrid)

        # variables for centre queries
        centre_queries_coords = []
        centre_queries_probs = []
        centre_queries_semantic_labels = []
        centre_queries_offsets = []
        centre_queries_batch_offsets = [0]

        #
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]
        total_inst_num = 0

        for i, idx in enumerate(id):
            xyz_origin, rgb, label, instance_label = self.val_data_files[idx]

            ### jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, True, True, True)
            # xyz_middle = self.dataAugment(xyz_origin, False, False, False)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            ### crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_origin = xyz_origin[valid_idxs]
            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), label.copy())
            instance_infos = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list
            inst_centres = inst_infos['instance_centre']  # (nInst, 3) (cx, cy, cz)
            inst_size = inst_infos['instance_size']
            inst_label = inst_infos['instance_label']

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            if 'Centre' in cfg.model_mode.split('_'):
                ### get instance centre heatmap
                grid_xyz = np.zeros((32 ** 3, 3), dtype=np.float32)
                grid_xyz += xyz_middle.min(axis=0, keepdims=True)
                grid_size = (xyz_middle.max(axis=0, keepdims=True) - xyz_middle.min(axis=0, keepdims=True)) / 32
                grid_xyz += grid_size / 2
                grid_xyz = grid_xyz.reshape(32, 32, 32, 3)
                for index in range(32):
                    grid_xyz[index, :, :, 0] = grid_xyz[index, :, :, 0] + index * grid_size[0, 0]
                    grid_xyz[:, index, :, 1] = grid_xyz[:, index, :, 1] + index * grid_size[0, 1]
                    grid_xyz[:, :, index, 2] = grid_xyz[:, :, index, 2] + index * grid_size[0, 2]
                grid_xyz = grid_xyz.reshape(-1, 3)

                ### size adaptive gaussian function or fixed sigma gaussian
                if not self.heatmap_sigma:
                    grid_infos = generate_adaptive_heatmap(
                        torch.tensor(grid_xyz, dtype=torch.float64), torch.tensor(inst_centres), torch.tensor(inst_size),
                        torch.tensor(inst_label), min_IoU=self.min_IoU, min_radius=np.linalg.norm(grid_size),
                    )
                    grid_centre_heatmap = grid_infos['heatmap']
                    grid_instance_label = grid_infos['grid_instance_label']
                    grid_instance_label[grid_instance_label == 20] = -100
                else:
                    grid_centre_heatmap = generate_heatmap(grid_xyz.astype(np.double), np.asarray(inst_centres),
                                                    sigma=self.heatmap_sigma)

                ### sample centre queries around instance centres
                centre_queries_coord = []
                centre_queries_offset = []
                n_queries = 100
                for inst_cent in inst_centres:
                    centre_queries_offset.append(torch.randn(n_queries, 3).double() * 0.05)
                    centre_query_coords = torch.from_numpy(inst_cent) - centre_queries_offset[-1]
                    centre_queries_coord.append(centre_query_coords)

                centre_queries_coord = torch.cat(centre_queries_coord, dim=0)
                centre_queries_offset = torch.cat(centre_queries_offset, dim=0)

                centre_queries_infos = generate_adaptive_heatmap(
                    centre_queries_coord, torch.tensor(inst_centres), torch.tensor(inst_size),
                    torch.tensor(inst_label), min_IoU=self.min_IoU, min_radius=np.linalg.norm(grid_size),
                )
                centre_queries_prob = centre_queries_infos['heatmap']
                centre_queries_semantic_label = centre_queries_infos['grid_instance_label']
                centre_queries_semantic_label[centre_queries_semantic_label == 20] = -100
                centre_queries_offset[centre_queries_semantic_label == -100] = torch.zeros_like(centre_queries_offset[0, :])

                centre_queries_batch_offsets.append(centre_queries_batch_offsets[-1] + centre_queries_coord.shape[0])

                norm_coords = normalize_3d_coordinate(
                    torch.cat((torch.from_numpy(xyz_middle), torch.from_numpy(np.asarray(inst_centres))), dim=0).unsqueeze(
                        dim=0).clone()
                )
                norm_inst_centres = norm_coords[:, -len(inst_centres):, :]
                grid_centre_gt = coordinate2index(norm_inst_centres, 32, coord_type='3d').squeeze()
                grid_centre_indicator = torch.cat(
                    (torch.LongTensor(grid_centre_gt.shape[0], 1).fill_(i), grid_centre_gt.unsqueeze(dim=-1)), dim=1
                )

                ### only the centre grid point require to predict the offset vector
                grid_cent_xyz = grid_xyz[grid_centre_gt]
                grid_centre_offset = grid_cent_xyz - np.array(inst_centres)
                grid_centre_offset = grid_centre_offset.squeeze()
                grid_centre_offset_label = torch.cat(
                    (torch.DoubleTensor(grid_centre_offset.shape[0], 1).fill_(i), torch.from_numpy(grid_centre_offset)), dim=1
                )

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            # variables for backbone
            point_locs.append(torch.cat((torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()), dim=1))  # (N, 4) (sample_index, xyz)
            point_coords.append(torch.from_numpy(np.concatenate((xyz_middle, xyz_origin), axis=1)))  # (N, 6) (shifted_xyz, original_xyz)
            point_feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)  # (N, 3) (rgb)
            # variables for point-wise predictions
            point_semantic_labels.append(torch.from_numpy(label))  # (N)
            point_instance_labels.append(torch.from_numpy(instance_label))  # (N)
            point_instance_infos.append(torch.from_numpy(instance_infos))  # (N, 9) (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            instance_centres.append(
                torch.cat(
                    (torch.DoubleTensor(len(inst_centres), 1).fill_(i), torch.from_numpy(np.array(inst_centres))),
                    dim=1
                )
            )  # (nInst, 4) (sample_index, instance_centre_xyz)

            if 'Centre' in cfg.model_mode.split('_'):
                # variables for grid-wise predictions
                grid_centre_heatmaps.append(grid_centre_heatmap.unsqueeze(dim=0))  # (B, nGrid)
                grid_centre_indicators.append(grid_centre_indicator)  # (NGrid, 2) (sample_index, grid_index)
                grid_centre_offset_labels.append(grid_centre_offset_label)  # (NGrid, 4) (sample_index, grid_centre_offset)
                grid_centre_semantic_labels.append(grid_instance_label.unsqueeze(dim=0))  # (B, nGrid)

                centre_queries_coords.append(centre_queries_coord)
                centre_queries_probs.append(centre_queries_prob)
                centre_queries_semantic_labels.append(centre_queries_semantic_label)
                centre_queries_offsets.append(centre_queries_offset)

            # variable for other uses
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)

        point_locs = torch.cat(point_locs, 0)  # (N) (sample_index)
        point_coords = torch.cat(point_coords, 0).to(torch.float32)  # (N, 6) (shifted_xyz, original_xyz)
        point_feats = torch.cat(point_feats, 0)  # (N, 3) (rgb)
        if cfg.pos_enc == 'XYZ':
            point_positional_encoding = torch.cat(
                (
                    torch.zeros(point_coords.shape[0], 1), self.positional_encoding_func(point_coords[:, :3], 5),
                    torch.zeros(point_coords.shape[0], 1)
                ), dim=1
            )
        elif cfg.pos_enc == 'XYZRGB':
            point_positional_encoding = torch.cat(
                (
                    torch.zeros(point_coords.shape[0], 1), self.positional_encoding_func(point_coords[:, :3], 3),
                    torch.zeros(point_coords.shape[0], 1), self.positional_encoding_func(point_feats, 2)
                ), dim=1
            )
        # variables for point-wise predictions
        point_semantic_labels = torch.cat(point_semantic_labels, 0).long()  # (N)
        point_instance_labels = torch.cat(point_instance_labels, 0).long()  # (N)
        point_instance_infos = torch.cat(point_instance_infos, 0).to(torch.float32) # (N, 9) (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_centres = torch.cat(instance_centres, 0)  # (nInst, 3) (instance_centre_xyz)

        # variables for backbone
        ret_dict['point_locs'] = point_locs  # (N, 4) (sample_index, xyz)
        ret_dict['point_coords'] = point_coords  # (N, 6) (shifted_xyz, original_xyz)
        ret_dict['point_feats'] = point_feats  # (N, 3) (rgb)
        ret_dict['point_positional_encoding'] = point_positional_encoding  # (N, 32) (0, xyz-18dim, 0, rgb-12dim)
        ret_dict['point_semantic_labels'] = point_semantic_labels  # (N)
        ret_dict['point_instance_labels'] = point_instance_labels  # (N)
        ret_dict['point_instance_infos'] = point_instance_infos  # (N, 9)
        ret_dict['instance_centres'] = instance_centres  # (nInst, 3) (instance_xyz)

        if 'Centre' in cfg.model_mode.split('_'):
            # variables for grid-wise predictions
            grid_centre_heatmaps = torch.cat(grid_centre_heatmaps, 0).to(torch.float32)  # (B, nGrid)
            grid_centre_indicators = torch.cat(grid_centre_indicators, 0)  # (NInst, 2) (sample_index, grid_index)
            grid_centre_offset_labels = torch.cat(grid_centre_offset_labels, 0).to(torch.float32)  # (NInst, 4) (sample_index, grid_centre_offset)
            grid_centre_semantic_labels = torch.cat(grid_centre_semantic_labels, 0)  # (B, nGrid)

            centre_queries_coords = torch.cat(centre_queries_coords, dim=0)
            centre_queries_probs = torch.cat(centre_queries_probs, dim=0).to(torch.float32)
            centre_queries_semantic_labels = torch.cat(centre_queries_semantic_labels, dim=0)
            centre_queries_offsets = torch.cat(centre_queries_offsets, dim=0).to(torch.float32)
            centre_queries_batch_offsets = torch.tensor(centre_queries_batch_offsets, dtype=torch.int)

            # variables for grid-wise predictions
            ret_dict['grid_centre_heatmap'] = grid_centre_heatmaps  # (B, nGrid)
            ret_dict['grid_centre_indicator'] = grid_centre_indicators  # (NGrid, 2) (sample_index, grid_index)
            ret_dict['grid_centre_offset_labels'] = grid_centre_offset_labels  # (NGrid, 4) (sample_index, grid_centre_offset)
            ret_dict['grid_centre_semantic_labels'] = grid_centre_semantic_labels  # (B, nGrid)

            ret_dict['centre_queries_coords'] = centre_queries_coords
            ret_dict['centre_queries_probs'] = centre_queries_probs
            ret_dict['centre_queries_semantic_labels'] = centre_queries_semantic_labels
            ret_dict['centre_queries_offsets'] = centre_queries_offsets
            ret_dict['centre_queries_batch_offsets'] = centre_queries_batch_offsets

        # variable for other uses
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int) # int (total_nInst)

        spatial_shape = np.clip((point_locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(point_locs, self.batch_size, self.mode)

        # variables for point-wise predictions
        ret_dict['voxel_locs'] = voxel_locs  # (nVoxel, 4)
        ret_dict['p2v_map'] = p2v_map  # (N)
        ret_dict['v2p_map'] = v2p_map  # (nVoxel, 19?)
        # variables for other uses
        ret_dict['instance_pointnum'] = instance_pointnum  # (nInst) # currently used in Jiang_PointGroup
        ret_dict['id'] = id
        ret_dict['batch_offsets'] = batch_offsets  # int (B+1)
        ret_dict['spatial_shape'] = spatial_shape  # long (3)


        if 'stuff' in cfg.model_mode.split('_'):
            nonstuff_spatial_shape = np.clip(
                (point_locs[point_semantic_labels > 1].max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)
            nonstuff_voxel_locs, nonstuff_p2v_map, nonstuff_v2p_map = pointgroup_ops.voxelization_idx(
                point_locs[point_semantic_labels > 1], self.batch_size, self.mode
            )

            # nonstuff related
            ret_dict['nonstuff_spatial_shape'] = nonstuff_spatial_shape
            ret_dict['nonstuff_voxel_locs'] = nonstuff_voxel_locs
            ret_dict['nonstuff_p2v_map'] = nonstuff_p2v_map
            ret_dict['nonstuff_v2p_map'] = nonstuff_v2p_map

        return ret_dict

    def testMerge(self, id):
        ret_dict = {}

        # variables for backbone
        point_locs = [] # (N, 4) (sample_index, xyz)
        point_coords = []  # (N, 6) (shifted_xyz, original_xyz)
        point_feats = []  # (N, 3) (rgb)
        point_semantic_labels = []
        point_instance_infos = []
        grid_xyzs = []

        batch_offsets = [0]
        total_inst_num = 0

        for i, idx in enumerate(id):
            if self.test_split == 'val':
                xyz_origin, rgb, label, instance_label = self.test_data_files[idx]
            elif self.test_split == 'test':
                xyz_origin, rgb = self.test_data_files[idx]
            else:
                print("Wrong test split: {}!".format(self.test_split))
                exit(0)

            ### jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            if 'Centre' in cfg.model_mode.split('_'):
                ### get instance center heatmap
                grid_xyz = np.zeros((32 ** 3, 3), dtype=np.float32)
                grid_xyz += xyz_middle.min(axis=0, keepdims=True)
                grid_size = (xyz_middle.max(axis=0, keepdims=True) - xyz_middle.min(axis=0, keepdims=True)) / 32
                grid_xyz += grid_size / 2
                grid_xyz = grid_xyz.reshape(32, 32, 32, 3)
                for index in range(32):
                    grid_xyz[index, :, :, 0] = grid_xyz[index, :, :, 0] + index * grid_size[0, 0]
                    grid_xyz[:, index, :, 1] = grid_xyz[:, index, :, 1] + index * grid_size[0, 1]
                    grid_xyz[:, :, index, 2] = grid_xyz[:, :, index, 2] + index * grid_size[0, 2]
                grid_xyz = grid_xyz.reshape(-1, 3)

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            # variables for backbone
            point_locs.append(torch.cat((torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()), dim=1))  # (N, 4) (sample_index, xyz)
            point_coords.append(torch.from_numpy(np.concatenate((xyz_middle, xyz_origin), axis=1)))  # (N, 6) (shifted_xyz, original_xyz)
            point_feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)  # (N, 3) (rgb)

            if 'Centre' in cfg.model_mode.split('_'):
                grid_xyzs.append(torch.from_numpy(grid_xyz))

            if self.test_split == 'val':
                point_semantic_labels.append(torch.from_numpy(label))  # (N)
                xyz, valid_idxs = self.crop(xyz)
                instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

                ### get instance information
                inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), label.copy())
                instance_infos = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
                point_instance_infos.append(torch.from_numpy(instance_infos))  # (N, 9) (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)

        point_locs = torch.cat(point_locs, 0)  # (N) (sample_index)
        point_coords = torch.cat(point_coords, 0).to(torch.float32)  # (N, 6) (shifted_xyz, original_xyz)
        point_feats = torch.cat(point_feats, 0)  # (N, 3) (rgb)

        if cfg.pos_enc == 'XYZ':
            point_positional_encoding = torch.cat(
                (
                    torch.zeros(point_coords.shape[0], 1), self.positional_encoding_func(point_coords[:, :3], 5),
                    torch.zeros(point_coords.shape[0], 1)
                ), dim=1
            )
        elif cfg.pos_enc == 'XYZRGB':
            point_positional_encoding = torch.cat(
                (
                    torch.zeros(point_coords.shape[0], 1), self.positional_encoding_func(point_coords[:, :3], 3),
                    torch.zeros(point_coords.shape[0], 1), self.positional_encoding_func(point_feats, 2)
                ), dim=1
            )

        # variables for backbone
        ret_dict['point_locs'] = point_locs  # (N, 4) (sample_index, xyz)
        ret_dict['point_coords'] = point_coords  # (N, 6) (shifted_xyz, original_xyz)
        ret_dict['point_feats'] = point_feats  # (N, 3) (rgb)
        ret_dict['point_positional_encoding'] = point_positional_encoding

        if 'Centre' in cfg.model_mode.split('_'):
            grid_xyzs = torch.cat(grid_xyzs, 0)

            ret_dict['grid_xyz'] = grid_xyzs

        if self.test_split == 'val':
            point_semantic_labels = torch.cat(point_semantic_labels, 0).long()  # (N)
            point_instance_infos = torch.cat(point_instance_infos, 0).to(torch.float32) # (N, 9) (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)

            ret_dict['point_semantic_labels'] = point_semantic_labels  # (N)
            ret_dict['point_instance_infos'] = point_instance_infos

        spatial_shape = np.clip((point_locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(point_locs, self.batch_size, self.mode)

        # variables for point-wise predictions
        ret_dict['voxel_locs'] = voxel_locs  # (nVoxel, 4)
        ret_dict['p2v_map'] = p2v_map  # (N)
        ret_dict['v2p_map'] = v2p_map  # (nVoxel, 19?)
        # variables for other uses
        ret_dict['id'] = id
        ret_dict['batch_offsets'] = batch_offsets  # int (B+1)
        ret_dict['spatial_shape'] = spatial_shape  # long (3)

        return ret_dict