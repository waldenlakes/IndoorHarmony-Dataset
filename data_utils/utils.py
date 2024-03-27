import os
import os.path as osp
import glob
import shutil
import random
import numpy as np
import cv2
import h5py
import time
from PIL import Image


def gamma_tonemapping(hdr, gamma=2.2, exposure=0.0):
    ldr = (2**exposure * hdr) ** (1 / gamma)
    return ldr 

def extract_bounding_box(fg_img, fg_mask):
    ret, bin_fg_mask = cv2.threshold((fg_mask*255).astype(np.uint8), 127, 1, cv2.THRESH_BINARY)
    # print(bin_fg_mask.dtype, bin_fg_mask.shape)
    points = cv2.boundingRect(bin_fg_mask)

    bb_fg_mask = fg_mask[points[1]:(points[1] + points[3]), points[0]:(points[0] + points[2])]
    bb_fg_img = fg_img[points[1]:(points[1] + points[3]), points[0]:(points[0] + points[2])]

    return bb_fg_img, bb_fg_mask

def PIL_resize_with_antialiasing(img, shape):
    img = np.clip(img, 0.0, 1.0)
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.resize(shape, Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)/255.0

    return img

def get_ratio_v2(bg_img_height, fg_img_height, placement_config):
    Pt_pixel_height_ = bg_img_height - int(placement_config['pt_h'] * bg_img_height)
    height_obj  = placement_config['Obj_height'] / (1 - placement_config['Obj_support_height']/placement_config['Cam_height'])
    tan_alpha = (bg_img_height*0.5 - Pt_pixel_height_) / (bg_img_height*0.5) * np.tan(placement_config["Cam_HFOV"]/2)
    Pt_pixel_height = bg_img_height*0.5 - (tan_alpha + np.tan(placement_config['Cam_Beta_angle'])) / \
        (1-tan_alpha*np.tan(placement_config['Cam_Beta_angle'])) * (bg_img_height*0.5) / np.tan(placement_config["Cam_HFOV"]/2)
    tan_phi = (bg_img_height/2-Pt_pixel_height) / (bg_img_height*0.5) * np.tan(placement_config["Cam_HFOV"]/2)
    tan_gamma = (height_obj - placement_config['Cam_height']) / placement_config['Cam_height'] * tan_phi
    OBJ_pixel_height = (tan_phi*np.tan(np.pi/2-placement_config["Cam_HFOV"]/2)) / \
        (1 - tan_phi*np.tan(np.pi/2-placement_config["Cam_HFOV"]/2))  * height_obj / placement_config['Cam_height'] * Pt_pixel_height \
         * ((tan_gamma+np.tan(placement_config['Cam_Beta_angle']))/(1-tan_gamma*np.tan(placement_config['Cam_Beta_angle']))+ \
            (tan_phi-np.tan(placement_config['Cam_Beta_angle']))/(1+tan_phi*np.tan(placement_config['Cam_Beta_angle']))) / (tan_gamma + tan_phi)
    RATIO = OBJ_pixel_height / fg_img_height

    return RATIO

def do_composition(raw_fg, raw_mask, bg_img, placement_config):
    '''
    raw_fg: [0, 1] np.float32
    raw_mask: [0, 1] np.float32 shape=(H, W)
    bg_img: [0, 1] np.float32
    placement_config: pt_h(0-1), pt_w(0-1), Obj_height(meter), Obj_support_height(meter), Cam_height(meter), Cam_HFOV(radian), Cam_Beta_angle(radian)
    '''
    bb_fg_img, bb_fg_mask = extract_bounding_box(raw_fg, raw_mask)

    # get ratio
    bg_img_height, fg_img_height = bg_img.shape[0], bb_fg_mask.shape[0]
    # RATIO = get_ratio(bg_img, bb_fg_mask, placement_config)
    RATIO = get_ratio_v2(bg_img_height, fg_img_height, placement_config)

    if RATIO <= 0:
        RATIO = 0.5
    bb_h, bb_w = bb_fg_mask.shape
    nh, nw = int(RATIO * bb_h), int(RATIO * bb_w)
    # reshaped_fg_img = cv2.resize(bb_fg_img, (nw, nh), cv2.INTER_AREA)#cv2.INTER_CUBIC)
    # reshaped_bb_fg_mask = cv2.resize(bb_fg_mask, (nw, nh), cv2.INTER_AREA)#cv2.INTER_CUBIC)
    reshaped_fg_img = PIL_resize_with_antialiasing(bb_fg_img, (nw, nh))
    reshaped_bb_fg_mask = PIL_resize_with_antialiasing(bb_fg_mask, (nw, nh))
    reshaped_bb_fg_mask = np.expand_dims(reshaped_bb_fg_mask, 2)

    h_index, w_index = int(placement_config['pt_h'] * bg_img.shape[0]) - nh, int(placement_config['pt_w'] * bg_img.shape[1]) - nw//2
    # checking boundary
    if h_index < 0:
        h_index = 0
    elif h_index > (bg_img.shape[0]-nh):
        h_index = bg_img.shape[0]-nh
    if w_index < 0:
        w_index = 0
    elif w_index > (bg_img.shape[1]-nw):
        w_index = bg_img.shape[1]-nw
    
    new_fg_mask = np.zeros_like(bg_img)
    new_fg_mask[h_index:h_index + nh, w_index:w_index + nw] = reshaped_bb_fg_mask
    new_fg_img = np.zeros_like(bg_img)
    new_fg_img[h_index:h_index + nh, w_index:w_index + nw] = reshaped_fg_img

    # new_fg_mask = new_fg_mask / 255.0
    composited = new_fg_mask * new_fg_img + (1 - new_fg_mask) * bg_img
    composited = np.clip(composited, 0.0, 1.0)


    return composited.astype(np.float32), new_fg_mask.astype(np.float32)
