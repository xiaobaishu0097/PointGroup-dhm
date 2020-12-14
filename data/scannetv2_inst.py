'''
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by Li Jiang
'''

import os, sys, glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import Dataset
import torch_scatter

from model.common import generate_heatmap, normalize_3d_coordinate, coordinate2index, generate_adaptive_heatmap

sys.path.append('../')

from util.config import cfg
from util.log import logger
from lib.pointgroup_ops.functions import pointgroup_ops

class ScannetDatast(Dataset):
    '''
    Dataset class for Scannet v2
    '''
    def __init__(self, data_mode='train'):
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

        self.data_mode = data_mode

        if self.data_mode == 'train':
            self.file_names = sorted(
                glob.glob(os.path.join(self.data_root, self.dataset, 'train', '*' + self.filename_suffix))
            )
            self.data_files = [torch.load(i) for i in self.file_names]
            # train_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*' + self.filename_suffix)))
            # self.data_files = [torch.load(train_file_names[0])]

            logger.info('Training samples: {}'.format(len(self.data_files)))

        elif self.data_mode == 'val':
            self.file_names = sorted(
                glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*' + self.filename_suffix))
            )
            self.data_files = [torch.load(i) for i in self.file_names]

            logger.info('Validation samples: {}'.format(len(self.data_files)))

        elif self.data_mode == 'test':
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1

            self.file_names = sorted(
                glob.glob(os.path.join(self.data_root, self.dataset, self.test_split, '*' + self.filename_suffix))
            )
            self.data_files = [torch.load(i) for i in self.file_names]
            # self.test_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*' + self.filename_suffix)))
            # self.test_files = [torch.load(self.test_file_names[0])]

            logger.info('Testing samples ({}): {}'.format(self.test_split, len(self.data_files)))

        self.data_max_length = 0
        for i in self.data_files:
            if self.data_max_length < len(i[0]):
                self.data_max_length = len(i[0])

    def __len__(self):
        return len(self.data_files)

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

    def __getitem__(self, idx):
        batch_offsets = [0]
        total_inst_num = 0

        if self.data_mode == 'train':
            xyz_origin, rgb, label, instance_label = self.data_files[idx]

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

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            # label[label == -100] = 20

            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), label.copy())
            instance_infos = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            instance_pointnum = inst_infos["instance_pointnum"]  # (nInst), list
            instance_centres = inst_infos['instance_centre']  # (nInst, 3) (cx, cy, cz)
            inst_size = inst_infos['instance_size']
            inst_label = inst_infos['instance_label']

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

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
                    torch.tensor(grid_xyz, dtype=torch.float64), torch.tensor(instance_centres), torch.tensor(inst_size),
                    torch.tensor(inst_label), min_IoU=self.min_IoU, min_radius=np.linalg.norm(grid_size),
                )
                instance_heatmap = grid_infos['heatmap']
                grid_instance_label = grid_infos['grid_instance_label']
                grid_instance_label[grid_instance_label == 20] = -100
            else:
                instance_heatmap = generate_heatmap(grid_xyz.astype(np.double), np.asarray(instance_centres),
                                                sigma=self.heatmap_sigma)

            norm_coords = normalize_3d_coordinate(
                torch.cat((torch.from_numpy(xyz_middle), torch.from_numpy(np.asarray(instance_centres))), dim=0).unsqueeze(
                    dim=0).clone()
            )
            norm_inst_centres = norm_coords[:, -len(instance_centres):, :]
            grid_centre_gt = coordinate2index(norm_inst_centres, 32, coord_type='3d')

            ### only the centre grid point require to predict the offset vector
            grid_cent_xyz = grid_xyz[grid_centre_gt]
            grid_centre_offset = grid_cent_xyz - np.array(instance_centres)
            grid_centre_offset = grid_centre_offset.squeeze()

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)

            locs = torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(0), torch.from_numpy(xyz).long()], 1) # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
            locs_float = torch.from_numpy(xyz_middle).to(torch.float32) # float (N, 3)
            feats = (torch.from_numpy(rgb) + torch.randn(3) * 0.1) # float (N, C)
            # feats = torch.from_numpy(rgb)
            labels = torch.from_numpy(label).long() # long (N)
            instance_labels = torch.from_numpy(instance_label).long() # long (N)

            instance_infos = torch.from_numpy(instance_infos).to(torch.float32)
            instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)
            instance_centres = torch.from_numpy(np.asarray(instance_centres)).to(torch.float32)

            instance_heatmap = instance_heatmap.to(torch.float32)
            grid_centre_gt = grid_centre_gt.squeeze().to(torch.float32)
            grid_centre_offset = torch.from_numpy(grid_centre_offset).to(torch.float32)
            grid_xyz = torch.from_numpy(grid_xyz)
            grid_instance_label = grid_instance_label.to(torch.float32)

            spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

            ### voxelize
            voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

            #TODO: uniform variable names
            return {
                'locs': locs,  # (N, 4)
                'voxel_locs': voxel_locs,  # (nVoxel, 4)
                'p2v_map': p2v_map,  # (N)
                'v2p_map': v2p_map,  # (nVoxel, 19?)
                'locs_float': locs_float,  # (N, 3)
                'feats': feats,  # (N, 3)
                'labels': labels,  # (N)
                'instance_labels': instance_labels,  # (N)
                'instance_info': instance_infos,  # (N, 9)
                'instance_pointnum': instance_pointnum,  # (nInst)
                'instance_centres': instance_centres,  # (nInst, 3)
                'instance_heatmap': instance_heatmap,  # (nGrid)
                'grid_centre_gt': grid_centre_gt,  # (nInst)
                'centre_offset_labels': grid_centre_offset,  # (nInst, 3)
                'grid_xyz': grid_xyz,  # (nGrid, 3)
                'centre_semantic_labels': grid_instance_label,  # (nGrid)
                'id': idx,
                'offsets': batch_offsets,  # int (B+1)
                'spatial_shape': spatial_shape,  # long (3)
            }

        elif self.data_mode == 'val':
            xyz_origin, rgb, label, instance_label = self.data_files[idx]

            ### jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)
            # xyz_middle = self.dataAugment(xyz_origin, False, False, False)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            # label[label == -100] = 20

            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32), label.copy())
            instance_infos = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            instance_pointnum = inst_infos["instance_pointnum"]  # (nInst), list
            instance_centres = inst_infos['instance_centre']  # (nInst, 3) (cx, cy, cz)
            inst_size = inst_infos['instance_size']
            inst_label = inst_infos['instance_label']

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

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
                    torch.tensor(grid_xyz, dtype=torch.float64), torch.tensor(instance_centres), torch.tensor(inst_size),
                    torch.tensor(inst_label), min_IoU=self.min_IoU, min_radius=np.linalg.norm(grid_size),
                )
                instance_heatmap = grid_infos['heatmap']
                grid_instance_label = grid_infos['grid_instance_label']
                grid_instance_label[grid_instance_label == 20] = -100
            else:
                instance_heatmap = generate_heatmap(grid_xyz.astype(np.double), np.asarray(instance_centres),
                                                    sigma=self.heatmap_sigma)

            norm_coords = normalize_3d_coordinate(
                torch.cat((torch.from_numpy(xyz_middle), torch.from_numpy(np.asarray(instance_centres))), dim=0).unsqueeze(
                    dim=0).clone()
            )
            norm_inst_centres = norm_coords[:, -len(instance_centres):, :]
            grid_centre_gt = coordinate2index(norm_inst_centres, 32, coord_type='3d')

            ### only the centre grid point require to predict the offset vector
            grid_cent_xyz = grid_xyz[grid_centre_gt]
            grid_centre_offset = grid_cent_xyz - np.array(instance_centres)
            grid_centre_offset = grid_centre_offset.squeeze()

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)

            locs = torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(0), torch.from_numpy(xyz).long()],
                             1)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
            locs_float = torch.from_numpy(xyz_middle).to(torch.float32)  # float (N, 3)
            feats = (torch.from_numpy(rgb) + torch.randn(3) * 0.1)  # float (N, C)
            # feats = torch.from_numpy(rgb)
            labels = torch.from_numpy(label).long()  # long (N)
            instance_labels = torch.from_numpy(instance_label).long()  # long (N)

            instance_infos = torch.from_numpy(instance_infos).to(torch.float32)
            instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)
            instance_centres = torch.from_numpy(np.asarray(instance_centres)).to(torch.float32)

            instance_heatmap = instance_heatmap.to(torch.float32)
            grid_centre_gt = grid_centre_gt.squeeze().to(torch.float32)
            grid_centre_offset = torch.from_numpy(grid_centre_offset).to(torch.float32)
            grid_xyz = torch.from_numpy(grid_xyz)
            grid_instance_label = grid_instance_label.to(torch.float32)

            spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

            ### voxelize
            voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

            return {
                'locs': locs,  # (N, 4)
                'voxel_locs': voxel_locs,  # (nVoxel, 4)
                'p2v_map': p2v_map,  # (N)
                'v2p_map': v2p_map,  # (nVoxel, 19?)
                'locs_float': locs_float,  # (N, 3)
                'feats': feats,  # (N, 3)
                'labels': labels,  # (N)
                'instance_labels': instance_labels,  # (N)
                'instance_info': instance_infos,  # (N, 9)
                'instance_pointnum': instance_pointnum,  # (nInst)
                'instance_centres': instance_centres,  # (nInst, 3)
                'instance_heatmap': instance_heatmap,  # (nGrid)
                'grid_centre_gt': grid_centre_gt,  # (nInst)
                'centre_offset_labels': grid_centre_offset,  # (nInst, 3)
                'grid_xyz': grid_xyz,  # (nGrid, 3)
                'centre_semantic_labels': grid_instance_label,  # (nGrid)
                'id': idx,
                'offsets': batch_offsets,  # int (B+1)
                'spatial_shape': spatial_shape,  # long (3)
            }

        elif self.data_mode == 'test':
            if self.test_split == 'test':
                xyz_origin, rgb = self.data_files[idx]
            else:
                print("Wrong test split: {}!".format(self.test_split))
                exit(0)

            ### flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)
            # xyz_middle = self.dataAugment(xyz_origin, False, False, False)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)

            locs = torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(0), torch.from_numpy(xyz).long()], 1)
            locs_float = torch.from_numpy(xyz_middle)
            feats = (torch.from_numpy(rgb) + torch.randn(3) * 0.1).to(torch.float64)

            spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

            ### voxelize
            voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

            return {
                'locs': locs,  # (N, 4)
                'voxel_locs': voxel_locs,  # (nVoxel, 4)
                'p2v_map': p2v_map,  # (N)
                'v2p_map': v2p_map,  # (nVoxel, 19?)
                'locs_float': locs_float,  # (N, 3)
                'feats': feats,  # (N, 3)
                'id': idx,
                'offsets': batch_offsets,  # int (B+1)
                'spatial_shape': spatial_shape,  # long (3)
            }
