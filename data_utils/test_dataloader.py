import os
import glob
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import torchvision.utils as utils

class TestIndoorHarmonyDataset(Dataset):
    def __init__(self, data_dir):
        super(TestIndoorHarmonyDataset, self).__init__()
        self.data_dir = data_dir
        self.background_images_dir = os.path.join(self.data_dir, 'background_imgs')
        self.gt_dir = os.path.join(self.data_dir, 'gt')
        self.mask_dir = os.path.join(self.data_dir, 'mask')
        self.input_dir = os.path.join(self.data_dir, 'input')

        self.unharmonized_img_paths = []
        self.gt_img_paths = []
        self.mask_img_paths = []
        self.bg_img_paths = []
        test_list_path = os.path.join(data_dir, "test_list.txt")
        for did in open(test_list_path):
            did = did.strip()
            self.unharmonized_img_paths.append(os.path.join(self.input_dir, did.split(' ')[0]+'.png'))
            self.gt_img_paths.append(os.path.join(self.gt_dir, did.split(' ')[0]+'.png'))
            self.mask_img_paths.append(os.path.join(self.mask_dir, did.split(' ')[0]+'.png'))
            self.bg_img_paths.append(os.path.join(self.background_images_dir, did.split(' ')[1]+'.png'))

    def __getitem__(self, index):

        # obtain the current paths 
        bg_img_path = self.bg_img_paths[index]
        input_img_path = self.unharmonized_img_paths[index]
        gt_image_path = self.gt_img_paths[index]
        mask_img_path = self.mask_img_paths[index]

        # read images
        bg_img = Image.open(bg_img_path)
        input_img = Image.open(input_img_path)
        gt_img = Image.open(gt_image_path)
        mask_img = Image.open(mask_img_path)

        input_img = np.array(input_img, dtype=np.float32)[:, :, :3] / 255.0
        mask_img = np.array(mask_img, dtype=np.float32)[:, :, :3] / 255.0
        gt_img = np.array(gt_img, dtype=np.float32)[:, :, :3] / 255.0
        bg_img = np.array(bg_img, dtype=np.float32)[:, :, :3] / 255.0

        # do composition
        comp = mask_img * input_img + (1 - mask_img) * bg_img
        gt = mask_img * gt_img + (1 - mask_img) * bg_img


        # convert all relevant tensors into pytorch tensors
        comp_tensor = torch.from_numpy(comp).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_img).permute(2, 0, 1)
        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1)
        bg_img_tensor = torch.from_numpy(bg_img).permute(2, 0, 1)

        return {
            'comp': comp_tensor,
            'mask': mask_tensor,
            'gt': gt_tensor,
            'bg_img': bg_img_tensor
        }

    def __len__(self):
        return len(self.gt_img_paths)

if __name__ == '__main__':
    # ------------------------- download the IndoorHarmony-Dataset and set the directory of test set -------------------------
    TEST_DATA_DIR = "./IndoorHarmonyDataset/test/"

    # ------------------------- init dataloader-------------------------
    dataset = TestIndoorHarmonyDataset(TEST_DATA_DIR)
    print(f'Number of test images: {len(dataset)}')
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            num_workers=2,
                                            batch_size=1,
                                            shuffle=False)

    # ------------------------- load data -------------------------
    for iter, tensors_dic in enumerate(test_loader):
        iter = iter + 1
        print(f'Test [{iter}|{len(test_loader)}]')

        # load input composite image, mask and gt
        comp = tensors_dic['comp']
        mask = tensors_dic['mask']
        gt = tensors_dic['gt']

        # TODO: TEST YOUR OWN MODEL
        