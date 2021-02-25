'''
Visualization
Written by Li Jiang
'''

import numpy as np
import mayavi.mlab as mlab
import os, argparse
import torch
from operator import itemgetter
from math import exp

from model.common import generate_adaptive_heatmap

COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

COLOR40 = np.array(
        [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168], [75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]])

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array(['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                        'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160]
}
SEMANTIC_IDX2NAME = {1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa', 7: 'table', 8: 'door', 9: 'window', 10: 'bookshelf', 11: 'picture',
                12: 'counter', 14: 'desk', 16: 'curtain', 24: 'refridgerator', 28: 'shower curtain', 33: 'toilet',  34: 'sink', 36: 'bathtub', 39: 'otherfurniture'}

SEMANTIC_NAME2INDEX = {
    'wall': 0, 'floor': 1, 'cabinet': 2, 'bed': 3, 'chair': 4, 'sofa': 5, 'table': 6, 'door': 7, 'window': 8,
    'bookshelf': 9, 'picture': 10, 'counter': 11, 'desk': 12, 'curtain': 13, 'refridgerator': 14,
    'shower curtain': 15, 'toilet': 16, 'sink': 17, 'bathtub': 18, 'otherfurniture': 19
}

def visualize_pts_rgb(fig, pts, rgb, scale=0.02, visual_text=None):
    pxs = pts[:, 0]
    pys = pts[:, 1]
    pzs = pts[:, 2]
    pt_colors = np.zeros((pxs.size, 4), dtype=np.uint8)
    pt_colors[:, 0:3] = rgb
    pt_colors[:, 3] = 255  # transparent

    scalars = np.arange(pxs.__len__())
    points = mlab.points3d(pxs, pys, pzs,  scalars,
                           mode='sphere',  # point sphere
                           # colormap='Accent',
                           scale_mode='vector',
                           scale_factor=scale,
                           figure=fig)
    if visual_text is not None:
        for text in visual_text:
            mlab.text3d(
                text['xyz'][0], text['xyz'][1], text['xyz'][2], text['semantic_str'],
                color=text['color'], scale=text['scale']
            )
    points.module_manager.scalar_lut_manager.lut.table = pt_colors

scaledGaussian = lambda x : exp(-(1/2)*((x/0.25)**2))

def get_coords_color(opt):
    input_file = os.path.join(opt.data_root, opt.room_split, opt.room_name + '_inst_nostuff.pth')
    assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
    if opt.room_split == 'test':
        xyz, rgb = torch.load(input_file)
    else:
        xyz, rgb, label, inst_label = torch.load(input_file)
    rgb = (rgb + 1) * 127.5
    visual_text = None

    if opt.task == 'offset_error':
        assert opt.room_split != 'test'
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

        pt_offsets_file = os.path.join(opt.result_root, opt.room_split, 'pt_offsets', opt.room_name + '.npy')
        assert os.path.isfile(pt_offsets_file), 'No point offset result - {}.'.format(pt_offsets_file)
        pt_offsets = np.load(pt_offsets_file)
        xyz = xyz + 1 * pt_offsets[:, :]

        pt_gt_offset_file = os.path.join(
            '../dataset/scannetv2/val_visualize/pt_instance_info',
            opt.room_name + '.npy'
        )
        assert os.path.isfile(pt_gt_offset_file), 'No point instance info - {}.'.format(pt_gt_offset_file)
        pt_gt_offsets = np.load(pt_gt_offset_file)

    elif opt.task == 'offset_pred':

        assert opt.room_split != 'test'
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

        pt_offsets_file = os.path.join(opt.result_root, opt.room_split, 'pt_offsets', opt.room_name + '.npy')
        assert os.path.isfile(pt_offsets_file), 'No grid points result - {}.'.format(pt_offsets_file)
        pt_offsets = np.load(pt_offsets_file)
        xyz = xyz + 1 * pt_offsets[:, :]

    elif opt.task == 'shifted_pred':

        assert opt.room_split != 'test'
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

        pt_shifted_coords_file = os.path.join(opt.result_root, opt.room_split, 'pt_shifted_coords', opt.room_name + '.npy')
        assert os.path.isfile(pt_shifted_coords_file), 'No grid points result - {}.'.format(pt_shifted_coords_file)
        pt_shifted_coords = np.load(pt_shifted_coords_file)
        xyz = pt_shifted_coords

    elif opt.task == 'rc_offset_pred':

        assert opt.room_split != 'test'
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

        pt_offsets_file = os.path.join(opt.result_root, opt.room_split, 'pt_offsets', opt.room_name + '.npy')
        assert os.path.isfile(pt_offsets_file), 'No grid points result - {}.'.format(pt_offsets_file)
        pt_offsets = np.load(pt_offsets_file)
        xyz = xyz + 1 * pt_offsets[:, :]

        remove_class = []
        for class_name in ['table', 'desk']:
            remove_class.append(SEMANTIC_NAME2INDEX[class_name])

        for class_idx in remove_class:
            valid_class_indx = (label != class_idx)
            xyz = xyz[valid_class_indx, :]
            rgb = rgb[valid_class_indx, :]
            label = label[valid_class_indx]
            inst_label = inst_label[valid_class_indx]

    elif opt.task == 'grid_gt':
        # sem_valid = (label != 100)
        # xyz = xyz[sem_valid]
        # rgb = rgb[sem_valid]
        # inst_label = inst_label[sem_valid]
        # label = label[sem_valid]

        assert opt.room_split != 'test'
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

        pt_offsets_file = os.path.join(opt.result_root, opt.room_split, 'pt_offsets', opt.room_name + '.npy')
        assert os.path.isfile(pt_offsets_file), 'No grid points result - {}.'.format(pt_offsets_file)
        pt_offsets = np.load(pt_offsets_file)
        xyz = xyz + 1 * pt_offsets[:, :]

        inst_centers = []
        inst_sizes = []
        for inst_id in np.unique(inst_label[object_idx]):
            inst_centers.append(xyz[inst_label == inst_id].mean(0))
            inst_sizes.append(xyz[inst_label == inst_id].max(0) - xyz[inst_label == inst_id].min(0))

        grid_xyz = np.zeros((32**3, 3), dtype=np.float32)
        grid_xyz += xyz.min(axis=0, keepdims=True)
        grid_size = (xyz.max(axis=0, keepdims=True) - xyz.min(axis=0, keepdims=True)) / 32
        grid_xyz += grid_size / 2
        grid_xyz = grid_xyz.reshape(32, 32, 32, 3)
        for i in range(32):
            grid_xyz[i, :, :, 0] = grid_xyz[i, :, :, 0] + i * grid_size[0, 0]
            grid_xyz[:, i, :, 1] = grid_xyz[:, i, :, 1] + i * grid_size[0, 1]
            grid_xyz[:, :, i, 2] = grid_xyz[:, :, i, 2] + i * grid_size[0, 2]
        grid_xyz = grid_xyz.reshape((-1, 3)) # , order='F'
        # for i in range(grid_xyz.shape[0]):
        #     grid_xyz[i, 0] = grid_xyz[i, 0] + grid_size[0, 0] * (i % 32)
        #     grid_xyz[i, 1] = grid_xyz[i, 1] + grid_size[0, 1] * ((i % 32**2) // 32)
        #     grid_xyz[i, 2] = grid_xyz[i, 2] + grid_size[0, 2] * (i // 32**2)

        # distance = torch.tensor(grid_xyz).unsqueeze(dim=1).repeat(1, len(inst_centers), 1) - torch.tensor(inst_centers).unsqueeze(dim=0).repeat(
        #     grid_xyz.shape[0], 1, 1)
        # gaussian_pro = torch.norm(distance, dim=2)
        # gaussian_pro = gaussian_pro.apply_(scaledGaussian)
        # gaussian_pro = gaussian_pro.max(dim=1)[0]
        # gaussian_pro[gaussian_pro < exp(-1)] = 0

        # gaussian_pro = generate_heatmap(grid_xyz, inst_centers, sigma=0.25)
        gaussian_pro = generate_adaptive_heatmap(
            torch.tensor(grid_xyz), torch.tensor(inst_centers), torch.tensor(inst_sizes),
            min_radius=np.linalg.norm(grid_size)
        )['heatmap']

        # norm_inst_centers = normalize_3d_coordinate(
        #     torch.cat((torch.from_numpy(xyz), torch.from_numpy(np.asarray(inst_centers))), dim=0).unsqueeze(
        #         dim=0).clone()
        # )[:, -len(inst_centers):, :]
        # gaussian_pro_index = coordinate2index(norm_inst_centers, 32, coord_type='3d')

        # gaussian_pro_index_file = os.path.join(opt.result_root, opt.room_split, 'grid_center_gt', opt.room_name + '.npy')
        # assert os.path.isfile(gaussian_pro_index_file), 'No grid points result - {}.'.format(gaussian_pro_index_file)
        # gaussian_pro_index = np.load(gaussian_pro_index_file)
        #
        # gaussian_pro = torch.zeros((32**3))
        # for i in gaussian_pro_index:
        #     gaussian_pro[i] = 1

        grid_rgb = np.ones((32 ** 3, 3)) * 255
        grid_rgb[:, 1] *= (1 - gaussian_pro).reshape(-1, ).numpy()
        grid_rgb[:, 2] *= (1 - gaussian_pro).reshape(-1, ).numpy()
        grid_rgb = grid_rgb.clip(0, 255)

        grid_xyz = grid_xyz[gaussian_pro.reshape(32 ** 3, ) != 0, :]
        grid_rgb = grid_rgb[gaussian_pro.reshape(32 ** 3, ) != 0, :]

        # grid_xyz = np.array(inst_centers)
        # grid_rgb = np.zeros_like(grid_xyz)

        # xyz = grid_xyz
        # rgb = grid_rgb
        xyz = np.concatenate((xyz, grid_xyz), axis=0)
        rgb = np.concatenate((rgb, grid_rgb), axis=0)

    elif opt.task == 'grid_pred':
        assert opt.room_split != 'test'
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb
        rgb = np.ones_like(rgb) * 255

        grid_center_preds_file = os.path.join(opt.result_root, opt.room_split, 'grid_center_preds', opt.room_name + '.npy')
        assert os.path.isfile(grid_center_preds_file), 'No grid points result - {}.'.format(grid_center_preds_file)
        grid_center_preds = np.load(grid_center_preds_file)
        # grid_center_preds[grid_center_preds < 0] = 0
        # grid_rgb = np.ones((32**3, 3))
        # grid_rgb[:, 0] = grid_rgb[:, 0]  * 220
        # grid_rgb = grid_rgb * grid_center_preds.reshape(-1, 1)
        # grid_xyz = np.zeros((32**3, 3), dtype=np.float32)
        # grid_xyz += xyz.min(axis=0, keepdims=True)
        # grid_size = (xyz.max(axis=0, keepdims=True) - xyz.min(axis=0, keepdims=True)) / 32
        # grid_xyz = grid_xyz.reshape(32, 32, 32, 3)
        # for i in range(32):
        #     grid_xyz[i, :, :, 0] = grid_xyz[i, :, :, 0] + i * grid_size[0, 0]
        #     grid_xyz[:, i, :, 1] = grid_xyz[:, i, :, 1] + i * grid_size[0, 1]
        #     grid_xyz[:, :, i, 2] = grid_xyz[:, :, i, 2] + i * grid_size[0, 2]
        # grid_xyz = grid_xyz.reshape(-1, 3)
        #
        # grid_xyz = grid_xyz[grid_center_preds.reshape(32 ** 3, ) != 0, :]
        # grid_rgb = grid_rgb[grid_center_preds.reshape(32 ** 3, ) != 0, :]
        #
        # xyz = np.concatenate((xyz, grid_xyz), axis=0)
        # rgb = np.concatenate((rgb, grid_rgb), axis=0)

        grid_xyz = grid_center_preds
        grid_rgb = np.ones_like(grid_center_preds)
        grid_rgb[:, 0] = grid_rgb[:, 0] * 220
        xyz = np.concatenate((xyz, grid_xyz), axis=0)
        rgb = np.concatenate((rgb, grid_rgb), axis=0)

    elif (opt.task == 'semantic_gt'):
        assert opt.room_split != 'test'
        label = label.astype(np.int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif (opt.task == 'instance_gt'):
        assert opt.room_split != 'test'
        inst_label = inst_label.astype(np.int)
        print("Instance number: {}".format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb

    elif (opt.task == 'semantic_pred'):
        assert opt.room_split != 'train'
        semantic_file = os.path.join(opt.result_root, opt.room_split, 'semantic_pred', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    elif (opt.task == 'occupancy'):
        assert opt.room_split != 'train'
        point_occupancy_preds_file = os.path.join(opt.result_root, opt.room_split, 'point_occupancy_preds', opt.room_name + '.npy')
        point_instance_labels_file = os.path.join(opt.result_root, opt.room_split, 'point_instance_labels', opt.room_name + '.npy')
        point_occupancy_labels_file = os.path.join(opt.result_root, opt.room_split, 'point_occupancy_labels', opt.room_name + '.npy')
        assert os.path.isfile(point_occupancy_preds_file), 'No semantic result - {}.'.format(point_occupancy_preds_file)
        assert os.path.isfile(point_instance_labels_file), 'No semantic result - {}.'.format(point_instance_labels_file)
        assert os.path.isfile(point_occupancy_labels_file), 'No semantic result - {}.'.format(point_occupancy_labels_file)
        point_occupancy_preds = np.load(point_occupancy_preds_file).astype(np.float)  # 0~19
        point_instance_labels = np.load(point_instance_labels_file).astype(np.float)  # 0~19
        point_occupancy_labels = np.load(point_occupancy_labels_file).astype(np.float)  # 0~19

        inst_label_rgb = np.zeros(rgb.shape)
        inst_label = inst_label.astype(np.int)
        object_idx = (inst_label >= 0)
        inst_label_rgb[object_idx] = COLOR20[inst_label[object_idx] % len(COLOR20)]
        rgb = inst_label_rgb
        rgb = np.ones_like(rgb)

        point_occupancy_labels[point_occupancy_labels == 0] = 1.1
        point_occupancy_error = np.abs(point_occupancy_preds.squeeze(axis=1) - np.log(point_occupancy_labels)) / np.log(point_occupancy_labels)
        rgb[inst_label > 1, 0] += (point_occupancy_error[inst_label > 1] * 2200)

    elif opt.task == 'semantic_error':
        assert opt.room_split != 'train'
        semantic_file = os.path.join(opt.result_root, opt.room_split, 'semantic_pred', opt.room_name + '.npy')
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19

        semantic_valid_indx = label > 1
        semantic_error_indx = np.zeros_like(label)
        semantic_error_indx[label_pred != label] = 1
        semantic_error_indx[~semantic_valid_indx] = 0
        semantic_error_indx = semantic_error_indx.astype(bool)

        rgb = np.zeros_like(rgb)
        rgb[semantic_error_indx, 0] = 255
        rgb[semantic_error_indx, 1] = 0
        rgb[semantic_error_indx, 2] = 0

        semantic_label_scale = (0.02, 0.02, 0.02)

        sem_error_inst_label = inst_label[semantic_error_indx]
        visual_text = []
        for inst in np.unique(sem_error_inst_label):
            inst_indx = (inst_label == inst)
            true_semantic_label = np.unique(label[inst_indx])
            assert len(true_semantic_label) == 1, 'single instance has multiple labels'
            true_semantic_label_str = SEMANTIC_NAMES[int(true_semantic_label)]

            false_semantic_label = np.unique(label_pred[inst_indx][label_pred[inst_indx] != label[inst_indx]])
            false_semantic_label = [SEMANTIC_NAMES[int(label_indx)] for label_indx in false_semantic_label]
            separator = ', '
            false_semantic_label_str = separator.join(false_semantic_label)

            inst_visual_label_str = 'True: {}; False: {}'.format(true_semantic_label_str, false_semantic_label_str)

            visual_text.append({
                'xyz':  np.mean(xyz[inst_indx], axis=0),
                'semantic_str': inst_visual_label_str,
                'color': (0, 1, 0),
                'scale': semantic_label_scale,
            })

    elif opt.task == 'instance_error':
        assert opt.room_split != 'train'

    elif (opt.task == 'instance_pred'):
        assert opt.room_split != 'train'
        instance_file = os.path.join(opt.result_root, opt.room_split, opt.room_name + '.txt')
        assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #
        for i in range(len(masks) - 1, -1, -1):
            mask_path = os.path.join(opt.result_root, opt.room_split, masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            if (float(masks[i][2]) < 0.09):
                continue
            mask = np.loadtxt(mask_path).astype(np.int)
            print('{} {}: {} pointnum: {}'.format(i, masks[i], SEMANTIC_IDX2NAME[int(masks[i][1])], mask.sum()))
            inst_label_pred_rgb[mask == 1] = COLOR20[i % len(COLOR20)]
        rgb = inst_label_pred_rgb

    if opt.room_split != 'test':
        sem_valid = (label != -100)
        if opt.task == 'grid_gt' or opt.task == 'grid_pred':
            sem_valid = np.concatenate((sem_valid, np.ones((grid_xyz.shape[0],), dtype=bool)))
        xyz = xyz[sem_valid]
        rgb = rgb[sem_valid]

    return xyz, rgb, visual_text




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='path to the input dataset files', default='../dataset/scannetv2')
    parser.add_argument('--result_root', help='path to the predicted results', default='exp/scannetv2/pointgroup/pointgroup_run1_scannet/result/epoch384_nmst0.3_scoret0.09_npointt100')
    parser.add_argument('--room_name', help='room_name', default='scene0000_00')
    parser.add_argument('--room_split', help='train / val / test', default='train')
    parser.add_argument('--task', help='input / semantic_gt / semantic_pred / instance_gt / instance_pred', default='input')
    opt = parser.parse_args()

    print(opt.room_name)

    xyz, rgb, visual_text = get_coords_color(opt)

    fig = mlab.figure(figure=None, bgcolor=(1.0, 1.0, 1.0), size=((800, 800)))
    visualize_pts_rgb(fig, xyz, rgb, visual_text=visual_text)
    mlab.show()

