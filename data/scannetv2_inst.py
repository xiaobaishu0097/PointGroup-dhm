'''
ScanNet v2 Dataloader (Modified from SparseConvNet Dataloader)
Written by Li Jiang
'''

import os, sys, glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import Dataset, DataLoader
import SharedArray as SA
import torch_scatter

from model.common import generate_heatmap, normalize_3d_coordinate, coordinate2index, generate_adaptive_heatmap

sys.path.append('../')

from lib.pointgroup_ops.functions import pointgroup_ops
from model.Pointnet2.pointnet2 import pointnet2_utils

SEMANTIC_NAME2INDEX = {
    'wall': 0, 'floor': 1, 'cabinet': 2, 'bed': 3, 'chair': 4, 'sofa': 5, 'table': 6, 'door': 7, 'window': 8,
    'bookshelf': 9, 'picture': 10, 'counter': 11, 'desk': 12, 'curtain': 13, 'refridgerator': 14,
    'shower curtain': 15, 'toilet': 16, 'sink': 17, 'bathtub': 18, 'otherfurniture': 19
}

class ScannetDatast:
    def __init__(self, cfg, test=False):
        if test:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1

        if cfg.instance_classifier['activate']:
            cfg.batch_size = 1

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

        self.dist = cfg.dist
        self.cache = cfg.cache

        self.model_mode = cfg.model_mode
        self.heatmap_sigma = cfg.heatmap_sigma
        self.min_IoU = cfg.min_IoU

        self.voxel_center_prediction = cfg.voxel_center_prediction

    def trainLoader(self):
        self.train_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'train', '*' + self.filename_suffix)))
        if not self.cache:
            self.train_files = [torch.load(i) for i in self.train_file_names]

        train_set = list(range(len(self.train_file_names)))
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if self.dist else None
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers,
                                            shuffle=(self.train_sampler is None), sampler=self.train_sampler, drop_last=True, pin_memory=True,
                                            worker_init_fn=self._worker_init_fn_)

    def valLoader(self):
        self.val_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*' + self.filename_suffix)))
        if not self.cache:
            self.val_files = [torch.load(i) for i in self.val_file_names]

        val_set = list(range(len(self.val_file_names)))
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_set) if self.dist else None
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers,
                                          shuffle=False, sampler=self.val_sampler, drop_last=False, pin_memory=True, worker_init_fn=self._worker_init_fn_)


    def testLoader(self):
        self.test_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, self.test_split, '*' + self.filename_suffix)))
        if not self.cache:
            self.test_files = [torch.load(i) for i in self.test_file_names]

        test_set = list(np.arange(len(self.test_file_names)))
        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.testMerge, num_workers=self.test_workers,
                                           shuffle=False, drop_last=False, pin_memory=True)

    def trainInstanceLoader(self, train_id):
        self.train_instance_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'train', '*' + self.filename_suffix)))
        self.train_instance_file_names = [self.train_instance_file_names[train_id]]
        if not self.cache:
            self.train_instance_files = [torch.load(i) for i in self.train_instance_file_names]

        train_set = list(range(len(self.train_instance_file_names)))
        self.train_instance_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if self.dist else None
        self.train_instance_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers,
                                            shuffle=(self.train_sampler is None), sampler=self.train_sampler, drop_last=True, pin_memory=True,
                                            worker_init_fn=self._worker_init_fn_)

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)

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
        instance_centers = []
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

            ### instance_centers
            instance_centers.append(mean_xyz_i)
            instance_sizes.append(size_xyz_i)
            if semantic_label is not None:
                semantic_label[semantic_label == -100] = 20
                instance_labels.append(np.argmax(np.bincount(np.int32(semantic_label[inst_idx_i]))))
                ret['instance_label'] = instance_labels

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

            ret['instance_info'] = instance_info
            ret['instance_pointnum'] = instance_pointnum
            ret['instance_center'] = instance_centers
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
        instance_centers = []  # (nInst, 3) (instance_xyz)
        instance_sizes = []

        #
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]
        total_inst_num = 0

        for i, idx in enumerate(id):
            if self.cache:
                fn = self.train_file_names[idx].split('/')[-1][:12]
                xyz_origin = SA.attach("shm://{}_xyz".format(fn)).copy()
                rgb = SA.attach("shm://{}_rgb".format(fn)).copy()
                label = SA.attach("shm://{}_label".format(fn)).copy()
                instance_label = SA.attach("shm://{}_instance_label".format(fn)).copy()
            else:
                xyz_origin, rgb, label, instance_label = self.train_files[idx]

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
            inst_centers = inst_infos['instance_center']  # (nInst, 3) (cx, cy, cz)
            inst_size = inst_infos['instance_size']
            inst_label = inst_infos['instance_label']

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

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
            instance_centers.append(
                torch.cat(
                    (torch.DoubleTensor(len(inst_centers), 1).fill_(i), torch.from_numpy(np.array(inst_centers))),
                    dim=1
                )
            )  # (nInst, 4) (sample_index, instance_center_xyz)
            instance_sizes.append(
                torch.cat(
                    (torch.DoubleTensor(torch.tensor(inst_size).shape[0], 1).fill_(i), torch.tensor(inst_size)),
                    dim=1
                )
            )

            # variable for other uses
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)

        point_locs = torch.cat(point_locs, 0)  # (N) (sample_index)
        point_coords = torch.cat(point_coords, 0).to(torch.float32)  # (N, 6) (shifted_xyz, original_xyz)
        point_feats = torch.cat(point_feats, 0)  # (N, 3) (rgb)
        # variables for point-wise predictions
        point_semantic_labels = torch.cat(point_semantic_labels, dim=0).long()  # (N)
        point_instance_labels = torch.cat(point_instance_labels, dim=0).long()  # (N)
        point_instance_infos = torch.cat(point_instance_infos, dim=0).to(torch.float32) # (N, 9) (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_centers = torch.cat(instance_centers, dim=0)  # (nInst, 3) (instance_center_xyz)
        instance_sizes = torch.cat(instance_sizes, dim=0)

        # variables for backbone
        ret_dict['point_locs'] = point_locs # (N, 4) (sample_index, xyz)
        ret_dict['point_coords'] = point_coords # (N, 6) (shifted_xyz, original_xyz)
        ret_dict['point_feats'] = point_feats # (N, 3) (rgb)
        ret_dict['point_semantic_labels'] = point_semantic_labels  # (N)
        ret_dict['point_instance_labels'] = point_instance_labels  # (N)
        ret_dict['point_instance_infos'] = point_instance_infos  # (N, 9)
        ret_dict['instance_centers'] = instance_centers  # (nInst, 3) (instance_xyz)
        ret_dict['instance_sizes'] = instance_sizes

        # variable for other uses
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int) # int (total_nInst)

        spatial_shape = np.clip((point_locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(point_locs, self.batch_size, self.mode)

        if 'occupancy' in self.model_mode.split('_'):
            voxel_instance_labels = point_instance_labels[v2p_map[:, 1].long()]
            ignore_voxel_index = (voxel_instance_labels == -100)
            voxel_instance_labels[ignore_voxel_index] = voxel_instance_labels.max() + 1
            num_instance_occupancy = torch_scatter.scatter_add(
                torch.ones_like(voxel_instance_labels), voxel_instance_labels
            )
            voxel_occupancy_labels = num_instance_occupancy[voxel_instance_labels]
            voxel_occupancy_labels[ignore_voxel_index] = 0
            voxel_instance_labels[ignore_voxel_index] = -100

            ret_dict['voxel_instance_labels'] = voxel_instance_labels
            ret_dict['voxel_occupancy_labels'] = voxel_occupancy_labels

        if self.voxel_center_prediction['activate']:
            voxel_coords = point_coords[v2p_map[:, 1].long(), :3]
            voxel_center_probs = []
            for batch_idx in range(self.batch_size):
                voxel_batch_idx = (voxel_locs[:, 0] == batch_idx)
                voxel_center_probability = generate_adaptive_heatmap(
                    voxel_coords[voxel_batch_idx].double(), instance_centers[instance_centers[:, 0] == batch_idx, 1:],
                    instance_sizes[instance_sizes[:, 0] == batch_idx, 1:], min_IoU=self.min_IoU,
                )['heatmap']
                voxel_center_probs.append(voxel_center_probability)
            voxel_center_probs_labels = torch.cat(voxel_center_probs).to(torch.float32)

            voxel_instance_info = point_instance_infos[v2p_map[:, 1].long(), :]
            voxel_center_offset_labels = voxel_instance_info[:, :3] - voxel_coords

            voxel_center_semantic_labels = point_semantic_labels[v2p_map[:, 1].long()]
            voxel_center_instance_labels = point_instance_labels[v2p_map[:, 1].long()]

            ret_dict['voxel_center_probs_labels'] = voxel_center_probs_labels
            ret_dict['voxel_center_offset_labels'] = voxel_center_offset_labels
            ret_dict['voxel_center_semantic_labels'] = voxel_center_semantic_labels
            ret_dict['voxel_center_instance_labels'] = voxel_center_instance_labels

        # variables for point-wise predictions
        ret_dict['voxel_locs'] = voxel_locs  # (nVoxel, 4)
        ret_dict['p2v_map'] = p2v_map  # (N)
        ret_dict['v2p_map'] = v2p_map  # (nVoxel, 19?)
        # variables for other uses
        ret_dict['instance_pointnum'] = instance_pointnum  # (nInst) # currently used in Jiang_PointGroup
        ret_dict['id'] = id
        ret_dict['batch_offsets'] = batch_offsets  # int (B+1)
        ret_dict['spatial_shape'] = spatial_shape  # long (3)

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
        instance_centers = []  # (nInst, 3) (instance_xyz)
        instance_sizes = []

        #
        instance_pointnum = []  # (total_nInst), int

        batch_offsets = [0]
        total_inst_num = 0

        for i, idx in enumerate(id):
            if self.cache:
                fn = self.val_file_names[idx].split('/')[-1][:12]
                xyz_origin = SA.attach("shm://{}_xyz".format(fn)).copy()
                rgb = SA.attach("shm://{}_rgb".format(fn)).copy()
                label = SA.attach("shm://{}_label".format(fn)).copy()
                instance_label = SA.attach("shm://{}_instance_label".format(fn)).copy()
            else:
                xyz_origin, rgb, label, instance_label = self.val_files[idx]

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
            inst_centers = inst_infos['instance_center']  # (nInst, 3) (cx, cy, cz)
            inst_size = inst_infos['instance_size']
            inst_label = inst_infos['instance_label']

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

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
            instance_centers.append(
                torch.cat(
                    (torch.DoubleTensor(len(inst_centers), 1).fill_(i), torch.from_numpy(np.array(inst_centers))),
                    dim=1
                )
            )  # (nInst, 4) (sample_index, instance_center_xyz)
            instance_sizes.append(
                torch.cat(
                    (torch.DoubleTensor(torch.tensor(inst_size).shape[0], 1).fill_(i), torch.tensor(inst_size)),
                    dim=1
                )
            )

            # variable for other uses
            instance_pointnum.extend(inst_pointnum)

        ### merge all the scenes in the batchd
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)

        point_locs = torch.cat(point_locs, 0)  # (N) (sample_index)
        point_coords = torch.cat(point_coords, 0).to(torch.float32)  # (N, 6) (shifted_xyz, original_xyz)
        point_feats = torch.cat(point_feats, 0)  # (N, 3) (rgb)
        # variables for point-wise predictions
        point_semantic_labels = torch.cat(point_semantic_labels, 0).long()  # (N)
        point_instance_labels = torch.cat(point_instance_labels, 0).long()  # (N)
        point_instance_infos = torch.cat(point_instance_infos, 0).to(torch.float32) # (N, 9) (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_centers = torch.cat(instance_centers, 0)  # (nInst, 3) (instance_center_xyz)
        instance_sizes = torch.cat(instance_sizes, dim=0)

        # variables for backbone
        ret_dict['point_locs'] = point_locs  # (N, 4) (sample_index, xyz)
        ret_dict['point_coords'] = point_coords  # (N, 6) (shifted_xyz, original_xyz)
        ret_dict['point_feats'] = point_feats  # (N, 3) (rgb)
        ret_dict['point_semantic_labels'] = point_semantic_labels  # (N)
        ret_dict['point_instance_labels'] = point_instance_labels  # (N)
        ret_dict['point_instance_infos'] = point_instance_infos  # (N, 9)
        ret_dict['instance_centers'] = instance_centers  # (nInst, 3) (instance_xyz)
        ret_dict['instance_sizes'] = instance_sizes

        # variable for other uses
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int) # int (total_nInst)

        spatial_shape = np.clip((point_locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(point_locs, self.batch_size, self.mode)

        if 'occupancy' in self.model_mode.split('_'):
            voxel_instance_labels = point_instance_labels[v2p_map[:, 1].long()]
            ignore_voxel_index = (voxel_instance_labels == -100)
            voxel_instance_labels[ignore_voxel_index] = voxel_instance_labels.max() + 1
            num_instance_occupancy = torch_scatter.scatter_add(
                torch.ones_like(voxel_instance_labels), voxel_instance_labels
            )
            voxel_occupancy_labels = num_instance_occupancy[voxel_instance_labels]
            voxel_occupancy_labels[ignore_voxel_index] = 0
            voxel_instance_labels[ignore_voxel_index] = -100

            ret_dict['voxel_instance_labels'] = voxel_instance_labels
            ret_dict['voxel_occupancy_labels'] = voxel_occupancy_labels

        if self.voxel_center_prediction['activate']:
            voxel_coords = point_coords[v2p_map[:, 1].long(), :3]
            voxel_center_probs = []
            for batch_idx in range(self.batch_size):
                voxel_batch_idx = (voxel_locs[:, 0] == batch_idx)
                voxel_center_probability = generate_adaptive_heatmap(
                    voxel_coords[voxel_batch_idx].double(), instance_centers[instance_centers[:, 0] == batch_idx, 1:],
                    instance_sizes[instance_sizes[:, 0] == batch_idx, 1:], min_IoU=self.min_IoU,
                )['heatmap']
                voxel_center_probs.append(voxel_center_probability)
            voxel_center_probs_labels = torch.cat(voxel_center_probs).to(torch.float32)

            voxel_instance_info = point_instance_infos[v2p_map[:, 1].long(), :]
            voxel_center_offset_labels = voxel_instance_info[:, :3] - voxel_coords

            voxel_center_semantic_labels = point_semantic_labels[v2p_map[:, 1].long()]
            voxel_center_instance_labels = point_instance_labels[v2p_map[:, 1].long()]

            ret_dict['voxel_center_probs_labels'] = voxel_center_probs_labels
            ret_dict['voxel_center_offset_labels'] = voxel_center_offset_labels
            ret_dict['voxel_center_semantic_labels'] = voxel_center_semantic_labels
            ret_dict['voxel_center_instance_labels'] = voxel_center_instance_labels

        # variables for point-wise predictions
        ret_dict['voxel_locs'] = voxel_locs  # (nVoxel, 4)
        ret_dict['p2v_map'] = p2v_map  # (N)
        ret_dict['v2p_map'] = v2p_map  # (nVoxel, 19?)
        # variables for other uses
        ret_dict['instance_pointnum'] = instance_pointnum  # (nInst) # currently used in Jiang_PointGroup
        ret_dict['id'] = id
        ret_dict['batch_offsets'] = batch_offsets  # int (B+1)
        ret_dict['spatial_shape'] = spatial_shape  # long (3)

        return ret_dict

    def testMerge(self, id):
        ret_dict = {}

        # variables for backbone
        point_locs = [] # (N, 4) (sample_index, xyz)
        point_coords = []  # (N, 6) (shifted_xyz, original_xyz)
        point_feats = []  # (N, 3) (rgb)
        point_semantic_labels = []
        point_instance_infos = []
        point_instance_labels = []  # (N)

        batch_offsets = [0]
        total_inst_num = 0

        for i, idx in enumerate(id):
            assert self.test_split in ['val', 'test']
            if not self.cache:
                if self.test_split == 'val':
                    xyz_origin, rgb, label, instance_label = self.test_files[idx]
                elif self.test_split == 'test':
                    xyz_origin, rgb = self.test_files[idx]
            else:
                fn = self.test_file_names[idx].split('/')[-1][:12]
                xyz_origin = SA.attach("shm://{}_xyz".format(fn)).copy()
                rgb = SA.attach("shm://{}_rgb".format(fn)).copy()

            ### jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])
            # variables for backbone
            point_locs.append(torch.cat((torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()), dim=1))  # (N, 4) (sample_index, xyz)
            point_coords.append(torch.from_numpy(np.concatenate((xyz_middle, xyz_origin), axis=1)))  # (N, 6) (shifted_xyz, original_xyz)
            point_feats.append(torch.from_numpy(rgb) + torch.randn(3) * 0.1)  # (N, 3) (rgb)
            if self.test_split == 'val':
                point_instance_labels.append(torch.from_numpy(instance_label))  # (N)

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
        if self.test_split == 'val':
            point_instance_labels = torch.cat(point_instance_labels, 0).long()  # (N)
        # variables for backbone
        ret_dict['point_locs'] = point_locs  # (N, 4) (sample_index, xyz)
        ret_dict['point_coords'] = point_coords  # (N, 6) (shifted_xyz, original_xyz)
        ret_dict['point_feats'] = point_feats  # (N, 3) (rgb)

        spatial_shape = np.clip((point_locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(point_locs, self.batch_size, self.mode)

        if ('occupancy' in self.model_mode.split('_')) and (self.test_split == 'val'):
            voxel_instance_labels = point_instance_labels[v2p_map[:, 1].long()]
            ignore_voxel_index = (voxel_instance_labels == -100)
            voxel_instance_labels[ignore_voxel_index] = voxel_instance_labels.max() + 1
            num_instance_occupancy = torch_scatter.scatter_add(
                torch.ones_like(voxel_instance_labels), voxel_instance_labels
            )
            voxel_occupancy_labels = num_instance_occupancy[voxel_instance_labels]
            voxel_occupancy_labels[ignore_voxel_index] = 0
            voxel_instance_labels[ignore_voxel_index] = -100

            ret_dict['voxel_instance_labels'] = voxel_instance_labels
            ret_dict['voxel_occupancy_labels'] = voxel_occupancy_labels

        # variables for point-wise predictions
        ret_dict['voxel_locs'] = voxel_locs  # (nVoxel, 4)
        ret_dict['p2v_map'] = p2v_map  # (N)
        ret_dict['v2p_map'] = v2p_map  # (nVoxel, 19?)
        # variables for other uses
        ret_dict['id'] = id
        ret_dict['batch_offsets'] = batch_offsets  # int (B+1)
        ret_dict['spatial_shape'] = spatial_shape  # long (3)

        return ret_dict