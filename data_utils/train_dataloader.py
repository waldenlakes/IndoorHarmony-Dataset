import os
import os.path as osp
import glob
from pickletools import optimize
import shutil
import random
import numpy as np
import cv2
import h5py
import time
from PIL import Image
from skimage.measure import block_reduce
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import torch
import torch
import torchvision.utils as utils
from torch.utils.data import Dataset
from torchvision import transforms

from utils import do_composition

class TrainIndoorHarmonyDataset(Dataset):
    def __init__(self, data_dir, num_of_uv=512, tgt_env_H_W=[16, 32], tgt_bg_H_W=[120, 160]):
        super(TrainIndoorHarmonyDataset, self).__init__()
        self.data_dir = data_dir
        self.replica_data_dir = osp.join(self.data_dir, 'Replica')
        self.outdoor_data_dir = osp.join(data_dir, "Outdoor")

        self.replica_bg_data_dir = osp.join(self.replica_data_dir, 'backgrounds')
        self.replica_lighting_dir = osp.join(self.replica_data_dir, 'per-pixel-lighting-16-32')
        self.tgt_env_H, self.tgt_env_W = tgt_env_H_W[0], tgt_env_H_W[1] 
        self.tgt_bg_H, self.tgt_bg_W = tgt_bg_H_W[0], tgt_bg_H_W[1]
        self.num_of_uv = num_of_uv

        self.FG_HEIGHT = {}
        for did in open(osp.join(self.data_dir, 'objs_height.txt')):
            did = did.strip()
            self.FG_HEIGHT[did.split(' ')[0]] = float(did.split(' ')[1])/100.0

        list_path = osp.join(self.replica_data_dir, f"train_list.txt")
        self.replica_line_infos = []
        for did in open(list_path):
            did = did.strip()
            self.replica_line_infos.append(did)

        # for laval data
        self.laval_data_dir = osp.join(self.data_dir, 'Laval')
        self.laval_bg_data_dir = osp.join(self.laval_data_dir, 'backgrounds')
        self.laval_lighting_dir = osp.join(self.laval_data_dir, 'hdr_images')


        laval_list_path = osp.join(self.laval_data_dir, f"train_list.txt")
        self.laval_line_infos = []
        for did in open(laval_list_path):
            did = did.strip()
            self.laval_line_infos.append(did)

    def __len__(self):
        return len(self.replica_line_infos) + len(self.laval_line_infos)

    def read_lightmask(self, lightmask_path):
        light_mask = (cv2.imread(lightmask_path) / 255.0)[..., 0].astype(np.int8)
        light_mask = light_mask.flatten()
        mask_indexs = np.where(light_mask==1)[0]

        return list(mask_indexs)

    def load_svhdr(self, scene_name, current_pano_id, bg_name, placement_config):
        current_bg_img_azimuth = bg_name.split('_')[2]
        svhdr_path = osp.join(self.replica_lighting_dir, scene_name, f"{current_pano_id}_{current_bg_img_azimuth}.h5")

        f = h5py.File(svhdr_path.replace('.exr', '.h5'), 'r')
        svhdr = f['data'][()]

        sampled_train_uv_index = []
        all_indexs = list(np.arange(120 * 160))
        mask_indexs = self.read_lightmask(osp.join(self.replica_lighting_dir, "lightmask", \
            scene_name, f"{current_pano_id}_{current_bg_img_azimuth}.bmp"))

        for item in mask_indexs:
            all_indexs.remove(item)

        random_indexs = np.random.choice(all_indexs, self.num_of_uv)
        for index in random_indexs:
            u, v = int(index / 160), index % 160
            sampled_train_uv_index.append(np.array([u, v], dtype=np.int32))

        sampled_train_uv_index = np.stack(sampled_train_uv_index, 0)
        sampled_svhdr = svhdr[sampled_train_uv_index[:,0], sampled_train_uv_index[:,1], ...]

        pt_h, pt_w = int(placement_config['pt_h'] * self.tgt_bg_H), int(placement_config['pt_w'] * self.tgt_bg_W)
        target_uv_index = np.array([[pt_h, pt_w]], dtype=np.int32)

        return sampled_svhdr, sampled_train_uv_index, svhdr[pt_h, pt_w], target_uv_index
        
    def get_placement_config(self, line_info, index):
        pt_h, pt_w = float(line_info.split(' ')[3+index].split('_')[1]) / 480, float(line_info.split(' ')[3+index].split('_')[2]) / 640
        Obj_height, Obj_support_height = self.FG_HEIGHT[line_info.split(' ')[0]], 0.0
        Cam_height, Cam_HFOV, Cam_Beta_angle = 1.5, 67.5/180*np.pi, -int(line_info.split(' ')[2].split('/')[1].split('_')[3])/180*np.pi

        return {
            'pt_h': pt_h, 'pt_w': pt_w,
            'Obj_height': Obj_height, 'Obj_support_height': Obj_support_height,
            'Cam_height': Cam_height, 'Cam_HFOV': Cam_HFOV, 'Cam_Beta_angle': Cam_Beta_angle,
        }


    def get_raw_unharmonized_path(self, OBJ, OBJ_ANGLE):
        unharmonized_scene_type = random.choice(['indoor', 'outdoor'])

        if unharmonized_scene_type == 'outdoor':
            outdoor_ILLUMs = os.listdir(osp.join(self.outdoor_data_dir, 'objects', OBJ, 'images'))

            unharmonized_fg_img_paths = []
            for illum in outdoor_ILLUMs:
                unharmonized_fg_img_paths += glob.glob(osp.join(self.outdoor_data_dir, 'objects', OBJ, 'images', illum, f'image_{OBJ_ANGLE}_*.png'))

            if len(unharmonized_fg_img_paths) == 0:
                unharmonized_fg_img_path = 'false'
            else:
                unharmonized_fg_img_path = random.choice(unharmonized_fg_img_paths)
                raw_un_fg = np.flip(cv2.imread(unharmonized_fg_img_path), -1).astype(np.float32) / 255.0


        if unharmonized_scene_type == 'indoor' or unharmonized_fg_img_path == 'false':
            indoor_ILLUMs = os.listdir(osp.join(self.replica_data_dir, 'objects', OBJ, 'images'))

            unharmonized_fg_img_paths = []
            for illum in indoor_ILLUMs:
                unharmonized_fg_img_paths += glob.glob(osp.join(self.replica_data_dir, 'objects', OBJ, 'images', illum, f'image_*_{OBJ_ANGLE}.png'))

            unharmonized_fg_img_path = random.choice(unharmonized_fg_img_paths)
            raw_un_fg = np.flip(cv2.imread(unharmonized_fg_img_path), -1).astype(np.float32) / 255.0

        return raw_un_fg

    def Replica_data(self, index):
        # obtain the current paths and meta-info
        line_info = self.replica_line_infos[index]
        OBJ, OBJ_ANGLE = line_info.split(' ')[0], line_info.split(' ')[1]
        bg_name, pano_name = line_info.split(' ')[2].split('/')[1], line_info.split(' ')[2].split('/')[0]
        scene_name, pano_id, bg_img_azimuth = pano_name[:-len(pano_name.split('_')[-1])-1], pano_name.split('_')[-1], int(bg_name.split('_')[2])
        Cam_theta, Cam_phi = bg_name.split('_')[2], bg_name.split('_')[3]

        random_pos = random.randint(0, len(line_info.split(' ')[3:])-1) # random selection from candidate loactions 
        vis_env_info = line_info.split(' ')[3+random_pos]
        vis_env_id, vis_cam_PtH, vis_cam_PtW = vis_env_info.split('_')[0], int(vis_env_info.split('_')[1]), int(vis_env_info.split('_')[2])
        fg_name = f"image_{scene_name.replace('_', '-')}_{pano_id}_background_2.2_{Cam_theta}_{Cam_phi}_90_{vis_cam_PtH}_{vis_cam_PtW}_{OBJ_ANGLE}"

        # obtain paths and read images
        raw_un_fg = self.get_raw_unharmonized_path(OBJ, OBJ_ANGLE)
        raw_mask_fore_path = osp.join(self.replica_data_dir, 'objects', OBJ, 'mask_foreground', f'image_{OBJ_ANGLE}.bmp')
        raw_gt_shading_path = osp.join(self.replica_data_dir, 'objects', OBJ, 'shading', f"{scene_name.replace('_', '-')}-{pano_id}", fg_name+'.png')
        raw_gt_color_path = osp.join(self.replica_data_dir, 'objects', OBJ, 'images', f"{scene_name.replace('_', '-')}-{pano_id}", fg_name+'.png')

        _, raw_mask = cv2.threshold(np.flip(cv2.imread(raw_mask_fore_path), -1), 10, 255, cv2.THRESH_BINARY)
        raw_mask = raw_mask.astype(np.float32) / 255.0
        raw_gt_shading_fg = np.flip(cv2.imread(raw_gt_shading_path), -1).astype(np.float32) / 255.0
        raw_gt_color = np.flip(cv2.imread(raw_gt_color_path), -1).astype(np.float32) / 255.0

        placement_config = self.get_placement_config(line_info, random_pos)

        # load data for shading net
        orig_bg_img = np.flip(cv2.imread(osp.join(self.replica_bg_data_dir, scene_name, f"{pano_id}_{bg_name}.png")), -1) / 255.0
        bg_img = cv2.resize(orig_bg_img, (self.tgt_bg_W, self.tgt_bg_H))
        sv_light_map, train_uv_index, target_light_map, target_uv_index = self.load_svhdr(scene_name, pano_id, bg_name, placement_config) # num_of_uv, 
    
        real, mask = do_composition(raw_gt_color, raw_mask[..., 0], orig_bg_img, placement_config)
        real_tensor = torch.from_numpy(real).permute(2, 0, 1)

        comp, mask = do_composition(raw_un_fg, raw_mask[..., 0], orig_bg_img, placement_config)
        comp_tensor = torch.from_numpy(comp).permute(2, 0, 1)

        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)

        # convert numpy to torch tensor
        svhdr_tensor = torch.from_numpy(sv_light_map.astype(np.float32)).permute(0, 3, 1, 2)
        train_uv_index = torch.from_numpy(train_uv_index).long()
        bg_img_tensor = torch.from_numpy(bg_img.astype(np.float32)).permute(2, 0, 1)
        orig_bg_img_tensor = torch.from_numpy(orig_bg_img.astype(np.float32)).permute(2, 0, 1)

        target_uv_index = torch.from_numpy(target_uv_index).long()
        raw_un_fg_tensor = torch.from_numpy(raw_un_fg).permute(2, 0, 1)
        raw_mask_tensor = torch.from_numpy(raw_mask).permute(2, 0, 1)
        target_light_map_tensor = torch.from_numpy(target_light_map).permute(2, 0, 1)
        raw_gt_shading_fg = torch.from_numpy(raw_gt_shading_fg).permute(2, 0, 1)
        raw_gt_color = torch.from_numpy(raw_gt_color).permute(2, 0, 1)

        return {
            'comp': comp_tensor,
            'gt': real_tensor,
            'mask': mask_tensor,
            'placement_config': placement_config,
            'bg_img': bg_img_tensor,
            'orig_bg_img': orig_bg_img_tensor,
            'svhdr': svhdr_tensor,
            'train_uv_index': train_uv_index,
            'filename': f"{scene_name}_{pano_id}_{bg_name.split('_')[2]}",
            'target_uv_index': target_uv_index,
            'tgt_env_map': target_light_map_tensor,
            'raw_un_fg': raw_un_fg_tensor,
            'raw_gt_shading': raw_gt_shading_fg,
            'raw_gt_color': raw_gt_color,
            'raw_bin_mask': raw_mask_tensor,
            'dataset_type': 'Replica'
        }

    def get_laval_raw_unharmonized_path(self, OBJ, OBJ_ANGLE):
        unharmonized_scene_type = random.choice(['indoor', 'outdoor'])

        if unharmonized_scene_type == 'outdoor':
            outdoor_ILLUMs = os.listdir(osp.join(self.outdoor_data_dir, 'objects', OBJ, 'images'))

            unharmonized_fg_img_paths = []
            for illum in outdoor_ILLUMs:
                unharmonized_fg_img_paths += glob.glob(osp.join(self.outdoor_data_dir, 'objects', OBJ, 'images', illum, f'image_{OBJ_ANGLE}_*.png'))

            if len(unharmonized_fg_img_paths) == 0:
                unharmonized_fg_img_path = 'false'
            else:
                unharmonized_fg_img_path = random.choice(unharmonized_fg_img_paths)
                raw_un_fg = np.flip(cv2.imread(unharmonized_fg_img_path), -1).astype(np.float32) / 255.0


        if unharmonized_scene_type == 'indoor' or unharmonized_fg_img_path == 'false':
            indoor_ILLUMs = os.listdir(osp.join(self.laval_data_dir, 'objects', OBJ, 'images'))

            unharmonized_fg_img_paths = []
            for illum in indoor_ILLUMs:
                unharmonized_fg_img_paths += glob.glob(osp.join(self.laval_data_dir, 'objects', OBJ, 'images', illum, f'image_*_{OBJ_ANGLE}.png'))

            unharmonized_fg_img_path = random.choice(unharmonized_fg_img_paths)
            raw_un_fg = np.flip(cv2.imread(unharmonized_fg_img_path), -1).astype(np.float32) / 255.0


        return raw_un_fg

    def Laval_data(self, index):
        # obtain the current paths and meta-info
        line_info = self.laval_line_infos[index]
        OBJ, OBJ_ANGLE = line_info.split(' ')[0], line_info.split(' ')[1]
        bg_name = line_info.split(' ')[2] + '.png'
        ILLUM = line_info.split(' ')[2].split('/')[0]
        # scene_name, pano_id, bg_img_azimuth = pano_name[:-len(pano_name.split('_')[-1])-1], pano_name.split('_')[-1], int(bg_name.split('_')[2])
        Cam_theta, Cam_phi = line_info.split(' ')[2].split('_')[2], bg_name.split('_')[3]

        random_pos = random.randint(0, len(line_info.split(' ')[3:])-1) # random selection from candidate loactions 

        # obtain paths and read images
        ILLUM_ANGLE = line_info.split(' ')[3+random_pos].split('_')[0]
        raw_un_fg = self.get_laval_raw_unharmonized_path(OBJ, OBJ_ANGLE)
        raw_mask_fore_path = osp.join(self.laval_data_dir, 'objects', OBJ, 'mask_foreground', f'image_{OBJ_ANGLE}.bmp')
        raw_gt_shading_path = glob.glob(osp.join(self.laval_data_dir, 'objects', OBJ, 'shading', ILLUM, f'image_{ILLUM}_{ILLUM_ANGLE}_*.png'))[0]
        raw_gt_color_path = glob.glob(osp.join(self.laval_data_dir, 'objects', OBJ, 'images', ILLUM, f'image_{ILLUM}_{ILLUM_ANGLE}_*.png'))[0]
        target_light_map_path = glob.glob(osp.join(self.laval_lighting_dir, ILLUM, f'{ILLUM}_{ILLUM_ANGLE}_*.exr'))[0]
        bg_img_path = osp.join(self.laval_bg_data_dir, bg_name)

        _, raw_mask = cv2.threshold(np.flip(cv2.imread(raw_mask_fore_path), -1), 10, 255, cv2.THRESH_BINARY)
        raw_mask = raw_mask.astype(np.float32) / 255.0
        raw_gt_shading_fg = np.flip(cv2.imread(raw_gt_shading_path), -1).astype(np.float32) / 255.0
        raw_gt_color = np.flip(cv2.imread(raw_gt_color_path), -1).astype(np.float32) / 255.0\
        
        # load data for shading net
        orig_bg_img = np.flip(cv2.imread(bg_img_path), -1) / 255.0
        bg_img = cv2.resize(orig_bg_img, (self.tgt_bg_W, self.tgt_bg_H))
            
        pt_h, pt_w = int(float(line_info.split(' ')[3+random_pos].split('_')[1])/4), int(float(line_info.split(' ')[3+random_pos].split('_')[2])/4)
        target_uv_index = np.array([[pt_h, pt_w]], dtype=np.int32)

        target_light_map = np.flip(cv2.imread(target_light_map_path, cv2.IMREAD_ANYCOLOR+cv2.IMREAD_ANYDEPTH), -1).astype(np.float32)
        target_light_map = cv2.resize(target_light_map, (self.tgt_env_W, self.tgt_env_H))


        Obj_height, Obj_support_height = self.FG_HEIGHT[OBJ], 0.0
        Cam_height, Cam_HFOV, Cam_Beta_angle = 1.5, 67.5/180*np.pi, -int(line_info.split(' ')[2].split('/')[1].split('_')[3])/180*np.pi
        placement_config =  {
            'pt_h': float(line_info.split(' ')[3+random_pos].split('_')[1])/480, 'pt_w': float(line_info.split(' ')[3+random_pos].split('_')[2])/640,
            'Obj_height': Obj_height, 'Obj_support_height': Obj_support_height,
            'Cam_height': Cam_height, 'Cam_HFOV': Cam_HFOV, 'Cam_Beta_angle': Cam_Beta_angle,
        }


        real, mask = do_composition(raw_gt_color, raw_mask[..., 0], orig_bg_img, placement_config)
        real_tensor = torch.from_numpy(real).permute(2, 0, 1)

        comp, mask = do_composition(raw_un_fg, raw_mask[..., 0], orig_bg_img, placement_config)
        comp_tensor = torch.from_numpy(comp).permute(2, 0, 1)

        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)

        # convert numpy to torch tensor
        bg_img_tensor = torch.from_numpy(bg_img.astype(np.float32)).permute(2, 0, 1)
        orig_bg_img_tensor = torch.from_numpy(orig_bg_img.astype(np.float32)).permute(2, 0, 1)

        target_uv_index = torch.from_numpy(target_uv_index).long()
        raw_un_fg_tensor = torch.from_numpy(raw_un_fg).permute(2, 0, 1)
        raw_mask_tensor = torch.from_numpy(raw_mask).permute(2, 0, 1)
        target_light_map_tensor = torch.from_numpy(target_light_map).permute(2, 0, 1)
        raw_gt_shading_fg = torch.from_numpy(raw_gt_shading_fg).permute(2, 0, 1)
        raw_gt_color = torch.from_numpy(raw_gt_color).permute(2, 0, 1)

        return {
            'comp': comp_tensor,
            'gt': real_tensor,
            'mask': mask_tensor,
            'placement_config': placement_config,
            'bg_img': bg_img_tensor,
            'orig_bg_img': orig_bg_img_tensor,
            'svhdr': 0,
            'train_uv_index': 0,
            'filename': f"{ILLUM}_{ILLUM_ANGLE}_{OBJ}_{OBJ_ANGLE}",
            'target_uv_index': target_uv_index,
            'tgt_env_map': target_light_map_tensor,
            'raw_un_fg': raw_un_fg_tensor,
            'raw_gt_shading': raw_gt_shading_fg,
            'raw_gt_color': raw_gt_color,
            'raw_bin_mask': raw_mask_tensor,
            'dataset_type': 'Laval'
        }

    def __getitem__(self, index):
        if index < len(self.laval_line_infos):
            sample =  self.Laval_data(index)
        else:
            sample =  self.Replica_data(index-len(self.laval_line_infos))

        return sample


if __name__ == '__main__':
    # ------------------------- download the IndoorHarmony-Dataset and set the directory of train set -------------------------
    TRAIN_DATA_DIR = "./IndoorHarmonyDataset/train"

    # ------------------------- init dataloader-------------------------
    dataset = TrainIndoorHarmonyDataset(TRAIN_DATA_DIR)
    print(f'Number of train images: {len(dataset)}')
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            num_workers=2,
                                            batch_size=1,
                                            shuffle=False)

    # ------------------------- load data -------------------------
    for iter, tensors_dic in enumerate(train_loader):
        iter = iter + 1
        print(f'Train [{iter}|{len(train_loader)}]')

        # load input composite image, mask and gt
        comp = tensors_dic['comp']
        mask = tensors_dic['mask']
        gt = tensors_dic['gt']

        # TODO: TRAIN YOUR OWN MODEL
        