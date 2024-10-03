import cv2
import math
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from .utils import bounding_box

####
def gen_colors(N, random_colors=True, bright=True):
    """
    Generate colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if random_colors:
        random.shuffle(colors)
    return np.array(colors) * 255


# def gen_colors_outline(N, outline_idx=2):
#     """
#     Generate colors.
#     To get visually distinct colors, generate them in HSV space then
#     convert to RGB.
#     Outline specific color
#     """
#     pallete_bright_1 = np.array([[255.0, 0.0, 0.0], # red
#                                 [204.0, 255.0, 0.0], # greeny yellow
#                                 [0.0, 255.0, 102.0], # green - Inflammatory
#                                 [0.0, 102.0, 255.0], # blue - Epithelial
#                                 [204.0, 0.0, 255.0]]) # pink

#     pallete_bright_2 = np.array([[255.0, 0.0, 0.0], # bright red
#                                 [255.0, 255.0, 0.0], # bright yellow
#                                 [0.0, 255.0, 0.0], # bright green
#                                 [0.0, 255.0, 255.0], # bright cyan
#                                 [0.0, 0.0, 255.0], # bright blue
#                                 [255.0, 0.0, 255.0]]) # bright pink

#     compare_pallete = np.array([[255.0, 0.0, 0.0], # bright red
#                                 [255.0, 255.0, 0.0], # bright yellow Neoplastic
#                                 [0.0, 255.0, 0.0], # bright green Inflammatory
#                                 [0.0, 255.0, 255.0], # bright cyan Connective
#                                 [255.0, 0.0, 255.0], # bright pink Dead cells
#                                 [0.0, 0.0, 255.0]]) # bright blue Epithelial


#     brightness = 0.6
#     hsv = [(i / N, 1, 1.0) if i == outline_idx else (i / N, 1, brightness) for i in range(N)]
#     colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
#     # return colors
#     return list(compare_pallete / 255.0)

####
def visualize_instances(mask, canvas=None, color_info=None, to_outline=None, skip=None):
    """
    Args:
        mask: array of NW
        color_info: tuple ((cfg.nuclei_type_dict, cfg.color_palete), pred_inst_type[:, None])
    Return:
        Image with the instance overlaid
    """

    canvas = (
        np.full(mask.shape + (3,), 200, dtype=np.uint8)
        if canvas is None
        else np.copy(canvas)
    )

    insts_list = list(np.unique(mask))  # [0,1,2,3,4,..,820]
    insts_list.remove(0)  # remove background

    if color_info is None:
        inst_colors = gen_colors(len(insts_list))

    if color_info is not None:
        unique_colors = {}
        for key in color_info[0][0].keys():  # type_dict
            if (bool(to_outline) is True) and (key != to_outline):
                unique_colors[color_info[0][0][key]] = [224.0, 224.0, 224.0]
            else:
                unique_colors[color_info[0][0][key]] = color_info[0][1][
                    key
                ]  # color palete

    for idx, inst_id in enumerate(insts_list):
        if (color_info[1][idx][0]) == 0:  # if background inst
            continue

        if (skip is not None) and (
            (color_info[1][idx][0]) in skip
        ):  # if we skip specific type
            continue

        if color_info is None:
            inst_color = inst_colors[idx]
        else:
            inst_color = unique_colors[int(color_info[1][idx][0])]

        inst_map = np.array(mask == inst_id, np.uint8)
        y1, y2, x1, x2 = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_canvas_crop = canvas[y1:y2, x1:x2]
        contours = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # For opencv-python >= 4.1.0.25
        cv2.drawContours(inst_canvas_crop, contours[0], -1, inst_color, 2)

        # cv2.drawContours(inst_canvas_crop, contours[1], -1, inst_color, 2)
        canvas[y1:y2, x1:x2] = inst_canvas_crop
    return canvas


####
def gen_figure(
    imgs_list,
    titles,
    fig_inch,
    shape=None,
    share_ax="all",
    show=False,
    colormap=plt.get_cmap("jet"),
):

    num_img = len(imgs_list)
    if shape is None:
        ncols = math.ceil(math.sqrt(num_img))
        nrows = math.ceil(num_img / ncols)
    else:
        nrows, ncols = shape

    # generate figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=share_ax, sharey=share_ax)
    axes = [axes] if nrows == 1 else axes

    # not very elegant
    idx = 0
    for ax in axes:
        for cell in ax:
            cell.set_title(titles[idx])
            cell.imshow(imgs_list[idx], cmap=colormap)
            cell.tick_params(
                axis="both",
                which="both",
                bottom="off",
                top="off",
                labelbottom="off",
                right="off",
                left="off",
                labelleft="off",
            )
            idx += 1
            if idx == len(titles):
                break
        if idx == len(titles):
            break

    fig.tight_layout()
    return fig


####


# Consolidated utils
import glob
import os
import json
import operator
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

def exp_decay_lr_schedule(nr_epochs, init_lr, decay_factor, decay_steps):
    lr_sched = []
    for i in range(1, nr_epochs):
        if i % decay_steps == 0:
            decay = decay_factor ** (i / decay_steps)
            new_lr = init_lr * decay
            lr_sched.append((i, new_lr))
    return lr_sched

def show_best_checkpoint(path, compare_value="epoch_num", comparator="max"):
    with open(os.path.join(path, "stats.json"), "r") as read_file:
        data = json.load(read_file)
        checkpoints = [epoch_stat[compare_value] for epoch_stat in data]
        if comparator is "max":
            chckp = max((checkpoints, i) for i, checkpoints in enumerate(checkpoints))
        elif comparator is "min":
            chckp = min((checkpoints, i) for i, checkpoints in enumerate(checkpoints))
        return (chckp, data)


def get_best_chkpts(path, metric_name, comparator=">"):
    """
    Return the best checkpoint according to some criteria.
    Note that it will only return valid path, so any checkpoint that has been
    removed wont be returned (i.e moving to next one that satisfies the criteria
    such as second best etc.)

    Args:
        path: directory contains all checkpoints, including the "stats.json" file
    """
    # info = []
    # for stat_file in glob.glob(f"{path}/*/*.json"):
    #     print (stat_file)
    #     stat = json.load(stat_file)
    #     info.append(stat)
    # print (info)

    stat_file = os.path.join(path, "stats.json")
    with open(stat_file) as f:
        info = json.load(f)

    ops = {
        ">": operator.gt,
        "<": operator.lt,
    }
    op_func = ops[comparator]

    if comparator == ">":
        best_value = -float("inf")
    else:
        best_value = +float("inf")

    best_chkpt = None
    for epoch_stat in info:
        epoch_value = epoch_stat[metric_name]
        if op_func(epoch_value, best_value):
            chkpt_path = os.path.join(
                path, "model-{}.index".format(epoch_stat["global_step"])
            )
            if os.path.isfile(chkpt_path):
                selected_stat = epoch_stat
                best_value = epoch_value
                best_chkpt = chkpt_path
    return best_chkpt, selected_stat


####
def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


####
def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


####
def cropping_center(x, crop_shape, batch=False):
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


####
def rm_n_mkdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


####
def get_files(data_dir_list, data_ext):
    """
    Given a list of directories containing data with extention 'data_ext',
    generate a list of paths for all files within these directories
    """
    data_files = []
    for sub_dir in data_dir_list:
        files_list = glob.glob("{}/*{}".format(sub_dir, data_ext))
        files_list.sort()  # ensure same order
        data_files.extend(files_list)
    return data_files


####
def get_inst_centroid(inst_map):
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


####
def show_np_array(array):
    plt.imshow(array)
    plt.show()
    plt.pause(5)
    plt.close()


# Consolidated patch extractor
import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils import cropping_center

#####
class PatchExtractor(object):
    """
    Extractor to generate patches with or without padding.
    Turn on debug mode to see how it is done.

    Args:
        x         : input image, should be of shape HWC
        win_size  : a tuple of (h, w)
        step_size : a tuple of (h, w)
        debug     : flag to see how it is done
    Return:
        a list of sub patches, each patch has dtype same as x

    Examples:
        >>> xtractor = PatchExtractor((450, 450), (120, 120))
        >>> img = np.full([1200, 1200, 3], 255, np.uint8)
        >>> patches = xtractor.extract(img, 'mirror')
    """

    def __init__(self, win_size, step_size, debug=False):

        self.patch_type = "mirror"
        self.win_size = win_size
        self.step_size = step_size
        self.debug = debug
        self.counter = 0

    def __get_patch(self, x, ptx):
        pty = (ptx[0] + self.win_size[0], ptx[1] + self.win_size[1])
        win = x[ptx[0] : pty[0], ptx[1] : pty[1]]
        assert (
            win.shape[0] == self.win_size[0] and win.shape[1] == self.win_size[1]
        ), "[BUG] Incorrect Patch Size {0}".format(win.shape)
        if self.debug:
            if self.patch_type == "mirror":
                cen = cropping_center(win, self.step_size)
                cen = cen[..., self.counter % 3]
                cen.fill(150)
            cv2.rectangle(x, ptx, pty, (255, 0, 0), 2)
            plt.imshow(x)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            self.counter += 1
        return win

    def __extract_valid(self, x):
        """
        Extracted patches without padding, only work in case win_size > step_size

        Note: to deal with the remaining portions which are at the boundary a.k.a
        those which do not fit when slide left->right, top->bottom), we flip
        the sliding direction then extract 1 patch starting from right / bottom edge.
        There will be 1 additional patch extracted at the bottom-right corner

        Args:
            x         : input image, should be of shape HWC
            win_size  : a tuple of (h, w)
            step_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x
        """

        im_h = x.shape[0]
        im_w = x.shape[1]

        def extract_infos(length, win_size, step_size):
            flag = (length - win_size) % step_size != 0
            last_step = math.floor((length - win_size) / step_size)
            last_step = (last_step + 1) * step_size
            return flag, last_step

        h_flag, h_last = extract_infos(im_h, self.win_size[0], self.step_size[0])
        w_flag, w_last = extract_infos(im_w, self.win_size[1], self.step_size[1])

        sub_patches = []
        #### Deal with valid block
        for row in range(0, h_last, self.step_size[0]):
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        #### Deal with edge case
        if h_flag:
            row = im_h - self.win_size[0]
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        if w_flag:
            col = im_w - self.win_size[1]
            for row in range(0, h_last, self.step_size[0]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        if h_flag and w_flag:
            ptx = (im_h - self.win_size[0], im_w - self.win_size[1])
            win = self.__get_patch(x, ptx)
            sub_patches.append(win)
        return sub_patches

    def __extract_mirror(self, x):
        """
        Extracted patches with mirror padding the boundary such that the
        central region of each patch is always within the orginal (non-padded)
        image while all patches' central region cover the whole orginal image

        Args:
            x         : input image, should be of shape HWC
            win_size  : a tuple of (h, w)
            step_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x
        """

        diff_h = self.win_size[0] - self.step_size[0]
        padt = diff_h // 2
        padb = diff_h - padt

        diff_w = self.win_size[1] - self.step_size[1]
        padl = diff_w // 2
        padr = diff_w - padl

        pad_type = "constant" if self.debug else "reflect"
        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), pad_type)
        sub_patches = self.__extract_valid(x)
        return sub_patches

    def extract(self, x, patch_type):
        patch_type = patch_type.lower()
        self.patch_type = patch_type
        if patch_type == "valid":
            return self.__extract_valid(x)
        elif patch_type == "mirror":
            return self.__extract_mirror(x)
        else:
            assert False, "Unknown Patch Type [%s]" % patch_type
        return


#####

###########################################################################

if __name__ == "__main__":
    # toy example for debug
    # 355x355, 480x480
    xtractor = PatchExtractor((450, 450), (120, 120), debug=True)
    a = np.full([1200, 1200, 3], 255, np.uint8)

    xtractor.extract(a, "mirror")
    xtractor.extract(a, "valid")


# Consolidated info
from tensorflow.train import AdamOptimizer as AdamOpt

MAP_TYPES = {
    "hv_consep": {
        "Inflammatory": 2,  # 1
        "Epithelial": 3,  # 2
        "Spindle": 4,  # 3
        "Misc": 1,  # 4
    },
    "hv_pannuke": {
        "Inflammatory": 1,  # 1
        "Epithelial": 4,  # 2
        "Neoplastic cells": 5,  # 3
        "Connective": 2,  # 4
        "Dead cells": 3,  # 5
    },
    "hv_monusac": {
        "Epithelial": 1, 
        "Lymphocyte": 2, 
        "Macrophage": 3, 
        "Neutrophil": 4,
    },
}

COLOR_PALETE = {
    "Inflammatory": (0.0, 255.0, 0.0),  # bright green
    "Dead cells": (255.0, 255.0, 0.0),  # bright yellow
    "Neoplastic cells": (255.0, 0.0, 0.0),  # red  # aka Epithelial malignant
    "Epithelial": (0.0, 0.0, 255.0),  # dark blue  # aka Epithelial healthy
    "Misc": (0.0, 0.0, 0.0),  # pure black  # aka 'garbage class'
    "Spindle": (0.0, 255.0, 255.0),  # cyan  # Fibroblast, Muscle and Endothelial cells
    "Connective": (0.0, 220.0, 220.0),  # darker cyan    # Connective plus Soft tissue cells
    "Background": (255.0, 0.0, 170.0),  # pink
    ###
    "Lymphocyte": (170.0, 255.0, 0.0),  # light green
    "Macrophage": (170.0, 0.0, 255.0),  # purple
    "Neutrophil": (255.0, 170.0, 0.0),  # orange
    "black": (32.0, 32.0, 32.0),  # black
}

# orignal size (win size) - input size - output size (step size)
# 540x540 - 270x270 - 80x80  hover
# 512x512 - 256x256 - 164x164 hover_opt
# DATA_CODE = {
#     'np_hv_opt': '512x512_164x164',
#     'np_hv'    : '540x540_80x80'
# }

MODEL_TYPES = {
    "hv_consep": "np_hv", 
    "hv_pannuke": "np_hv_opt", 
    "hv_monusac": "np_hv_opt"
}

MODEL_PARAMS = {
    "np_hv": 
    {
        "step_size": (80,80),
        "win_size": (540,540),

        "train_input_shape": (270, 270),
        "train_mask_shape": (80, 80),
        "infer_input_shape": (270, 270),
        "infer_mask_shape": (80, 80),
        "training_phase": [
            {
                "nr_epochs": 50,
                "manual_parameters": {
                    # tuple(initial value, schedule)
                    "learning_rate": (1.0e-4, [("25", 1.0e-5)]),
                },
                "pretrained_path": "/data/input/pretrained/ImageNet-ResNet50-Preact.npz",
                "train_batch_size": 8,
                "infer_batch_size": 16,
                "model_flags": {"freeze": True},
            },
            {
                "nr_epochs": 50,
                "manual_parameters": {
                    # tuple(initial value, schedule)
                    "learning_rate": (1.0e-4, [("25", 1.0e-5)]),
                },
                # path to load, -1 to auto load checkpoint from previous phase,
                # None to start from scratch
                "pretrained_path": -1,
                "train_batch_size": 4,  # unfreezing everything will
                "infer_batch_size": 16,
                "model_flags": {"freeze": False},
            },
        ],
        "loss_term": {"bce": 1, "dice": 1, "mse": 2, "msge": 1},
        "optimizer": AdamOpt,
    },

    "np_hv_opt": 
    {
        "step_size": (164,164),
        "win_size": (512,512),

        "train_input_shape": (256, 256),
        "train_mask_shape": (164, 164),
        "infer_input_shape": (256, 256),
        "infer_mask_shape": (164, 164),
        "training_phase": [
            {
                "nr_epochs": 50,
                "manual_parameters": {
                    # tuple(initial value, schedule)
                    "learning_rate": (1.0e-4, [("25", 1.0e-5)]),
                },
                "pretrained_path": "/data/input/pretrained/ImageNet-ResNet50-Preact.npz",
                "train_batch_size": 8,
                "infer_batch_size": 16,
                "model_flags": {"freeze": True},
            },
            {
                "nr_epochs": 50,
                "manual_parameters": {
                    # tuple(initial value, schedule)
                    "learning_rate": (1.0e-4, [("25", 1.0e-5)]),
                },
                # path to load, -1 to auto load checkpoint from previous phase,
                # None to start from scratch
                "pretrained_path": -1,
                "train_batch_size": 4,  # unfreezing everything will
                "infer_batch_size": 16,
                "model_flags": {"freeze": False},
            },
        ],
        "loss_term": {"bce": 1, "dice": 1, "mse": 2, "msge": 1},
        "optimizer": AdamOpt,
    }
}

#### Training parameters
###
# np+hv : double branches nework,
#     1 branch nuclei pixel classification (segmentation)
#     1 branch regressing horizontal/vertical coordinate w.r.t the (supposed)
#     nearest nuclei centroids, coordinate is normalized to 0-1 range
#
# np+dst: double branches nework
#     1 branch nuclei pixel classification (segmentation)
#     1 branch regressing nuclei instance distance map (chessboard in this case),
#     the distance map is normalized to 0-1 range

# STEP_SIZE = {"hv_consep": (80,80), "hv_pannuke": (164,164), "hv_monusac": (164,164)}
# WIN_SIZE = {"hv_consep": (540,540), "hv_pannuke": (512,512), "hv_monusac": (512,512)}
# INPUT_SHAPE = { # WIN/2
#     'hv_consep': 270,
#     'hv_pannuke': 256,
#     'hv_monusac': 256,
# }
# MASK_SHAPE =  { # = STEP_SIZE
#     'hv_consep': 80,
#     'hv_pannuke': 164
#     'hv_monusac': 164
# }