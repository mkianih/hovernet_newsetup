import random
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tensorpack.dataflow import (
    AugmentImageComponent,
    AugmentImageComponents,
    BatchData,
    BatchDataByShape,
    CacheData,
    PrefetchDataZMQ,
    RNGDataFlow,
    RepeatedData,
)

####
class DatasetSerial(RNGDataFlow):
    """
    Produce ``(image, label)`` pair, where
        ``image`` has shape HWC and is RGB, has values in range [0-255].

        ``label`` is a float image of shape (H, W, C). Number of C depends
                  on `self.model_mode` within `config.py`

                  If self.model_mode is 'np+xy':
                    channel 0 binary nuclei map, values are either 0 (background) or 1 (nuclei)
                    channel 1 containing the X-map, values in range [-1, 1]
                    channel 2 containing the Y-map, values in range [-1, 1]

                  If self.model_mode is 'np+dst':
                    channel 0 binary nuclei map, values are either 0 (background) or 1 (nuclei)
                    channel 1 containing the per nuclei distance map, values in range [0, 1]
    """

    def __init__(self, path_list):
        self.path_list = path_list

    ##
    def size(self):
        return len(self.path_list)

    ##
    def get_data(self):
        idx_list = list(range(0, len(self.path_list)))
        random.shuffle(idx_list)
        for idx in idx_list:

            data = np.load(self.path_list[idx])

            # split stacked channel into image and label
            img = data[..., :3]  # RGB images
            ann = data[..., 3:]  # instance ID map
            # TODO: assert to ensure correct dtype

            img = img.astype("uint8")
            yield [img, ann]


####
def valid_generator(
    ds, shape_aug=None, input_aug=None, label_aug=None, batch_size=16, nr_procs=1
):
    ### augment both the input and label
    ds = (
        ds
        if shape_aug is None
        else AugmentImageComponents(ds, shape_aug, (0, 1), copy=True)
    )
    ### augment just the input
    ds = (
        ds
        if input_aug is None
        else AugmentImageComponent(ds, input_aug, index=0, copy=False)
    )
    ### augment just the output
    ds = (
        ds
        if label_aug is None
        else AugmentImageComponent(ds, label_aug, index=1, copy=True)
    )
    #
    ds = BatchData(ds, batch_size, remainder=True)
    ds = CacheData(ds)  # cache all inference images
    return ds


####
def train_generator(
    ds, shape_aug=None, input_aug=None, label_aug=None, batch_size=16, nr_procs=8
):
    ### augment both the input and label
    ds = (
        ds
        if shape_aug is None
        else AugmentImageComponents(ds, shape_aug, (0, 1), copy=True)
    )
    ### augment just the input i.e index 0 within each yield of DatasetSerial
    ds = (
        ds
        if input_aug is None
        else AugmentImageComponent(ds, input_aug, index=0, copy=False)
    )
    ### augment just the output i.e index 1 within each yield of DatasetSerial
    ds = (
        ds
        if label_aug is None
        else AugmentImageComponent(ds, label_aug, index=1, copy=True)
    )
    #
    ds = BatchDataByShape(ds, batch_size, idx=0)
    ds = PrefetchDataZMQ(ds, nr_procs)
    return ds


####
def visualize(datagen, batch_size, view_size=4, aug_only=False, preview=False):
    """
    Read the batch from 'datagen' and display 'view_size' number of
    of images and their corresponding Ground Truth
    """

    def prep_imgs(img, ann):
        cmap = plt.get_cmap("viridis")
        # cmap may randomly fails if of other types
        ann = ann.astype("float32")
        ann_chs = np.dsplit(ann, ann.shape[-1])
        for i, ch in enumerate(ann_chs):
            ch = np.squeeze(ch)
            # normalize to -1 to 1 range else
            # cmap may behave stupidly
            ch = ch / (np.max(ch) - np.min(ch) + 1.0e-16)
            # take RGB from RGBA heat map
            ann_chs[i] = cmap(ch)[..., :3]
        img = img.astype("float32") / 255.0
        prepped_img = np.concatenate([img] + ann_chs, axis=1)
        return prepped_img

    assert view_size <= batch_size, "Number of displayed images must <= batch size"
    ds = RepeatedData(datagen, -1)
    ds.reset_state()
    for imgs, segs in ds.get_data():
        for idx in range(0, view_size):
            displayed_img = prep_imgs(imgs[idx], segs[idx])
            plt.subplot(view_size, 1, idx + 1)
            if aug_only:
                plt.imshow(imgs[idx])  # displayed_img
            else:
                plt.imshow(displayed_img)
        plt.savefig(f"{str(tempfile.NamedTemporaryFile().name)}.png")
        plt.show()
        if preview:
            break

    return


###########################################################################


# Consolidated augmentations
import math
import random

import cv2
import matplotlib.cm as cm
import numpy as np

from scipy import ndimage
from scipy.ndimage import measurements
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import affine_transform, map_coordinates
from scipy.ndimage.morphology import distance_transform_cdt, distance_transform_edt
from skimage import morphology as morph
from skimage import img_as_ubyte
from skimage.color import rgb2hed, hed2rgb, rgb2gray, gray2rgb, rgb2hsv
from skimage.exposure import equalize_hist, rescale_intensity

from matplotlib.colors import LinearSegmentedColormap

from tensorpack.dataflow.imgaug import ImageAugmentor
from tensorpack.utils.utils import get_rng

from misc.utils import cropping_center, bounding_box

####
class GenInstance(ImageAugmentor):
    def __init__(self, crop_shape=None):
        super(GenInstance, self).__init__()
        self.crop_shape = crop_shape

    def reset_state(self):
        self.rng = get_rng(self)

    def _fix_mirror_padding(self, ann):
        """
        Deal with duplicated instances due to mirroring in interpolation
        during shape augmentation (scale, rotation etc.)
        """
        current_max_id = np.amax(ann)
        inst_list = list(np.unique(ann))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(ann == inst_id, np.uint8)
            remapped_ids = measurements.label(inst_map)[0]
            remapped_ids[remapped_ids > 1] += current_max_id
            ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
            current_max_id = np.amax(ann)
        return ann


####
import matplotlib.pyplot as plt


class GenInstanceUnetMap(GenInstance):
    """
    Input annotation must be of original shape.

    Perform following operation:
        1) Remove the 1px of boundary of each instance
           to create separation between touching instances
        2) Generate the weight map from the result of 1)
           according to the unet paper equation.

    Args:
        wc (dict)        : Dictionary of weight classes.
        w0 (int/float)   : Border weight parameter.
        sigma (int/float): Border width parameter.
    """

    def __init__(self, wc=None, w0=10.0, sigma=5.0, crop_shape=None):
        super(GenInstanceUnetMap, self).__init__()
        self.crop_shape = crop_shape
        self.wc = wc
        self.w0 = w0
        self.sigma = sigma

    def _remove_1px_boundary(self, ann):
        new_ann = np.zeros(ann.shape[:2], np.int32)
        inst_list = list(np.unique(ann))
        inst_list.remove(0)  # 0 is background

        k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

        for idx, inst_id in enumerate(inst_list):
            inst_map = np.array(ann == inst_id, np.uint8)
            inst_map = cv2.erode(inst_map, k, iterations=1)
            new_ann[inst_map > 0] = inst_id
        return new_ann

    def _get_weight_map(self, ann, inst_list):
        if len(inst_list) <= 1:  # 1 instance only
            return np.zeros(ann.shape[:2])
        stacked_inst_bgd_dst = np.zeros(ann.shape[:2] + (len(inst_list),))

        for idx, inst_id in enumerate(inst_list):
            inst_bgd_map = np.array(ann != inst_id, np.uint8)
            inst_bgd_dst = distance_transform_edt(inst_bgd_map)
            stacked_inst_bgd_dst[..., idx] = inst_bgd_dst

        near1_dst = np.amin(stacked_inst_bgd_dst, axis=2)
        near2_dst = np.expand_dims(near1_dst, axis=2)
        near2_dst = stacked_inst_bgd_dst - near2_dst
        near2_dst[near2_dst == 0] = np.PINF  # very large
        near2_dst = np.amin(near2_dst, axis=2)
        near2_dst[ann > 0] = 0  # the instances
        near2_dst = near2_dst + near1_dst
        # to fix pixel where near1 == near2
        near2_eve = np.expand_dims(near1_dst, axis=2)
        # to avoide the warning of a / 0
        near2_eve = (1.0 + stacked_inst_bgd_dst) / (1.0 + near2_eve)
        near2_eve[near2_eve != 1] = 0
        near2_eve = np.sum(near2_eve, axis=2)
        near2_dst[near2_eve > 1] = near1_dst[near2_eve > 1]
        #
        pix_dst = near1_dst + near2_dst
        pen_map = pix_dst / self.sigma
        pen_map = self.w0 * np.exp(-(pen_map ** 2) / 2)
        pen_map[ann > 0] = 0  # inner instances zero
        return pen_map

    def _augment(self, img, _):
        img = np.copy(img)
        orig_ann = img[..., 0]  # instance ID map
        fixed_ann = self._fix_mirror_padding(orig_ann)
        # setting 1 boundary pix of each instance to background
        fixed_ann = self._remove_1px_boundary(fixed_ann)

        # cant do the shortcut because near2 also needs instances
        # outside of cropped portion
        inst_list = list(np.unique(fixed_ann))
        inst_list.remove(0)  # 0 is background
        wmap = self._get_weight_map(fixed_ann, inst_list)

        if self.wc is None:
            wmap += 1  # uniform weight for all classes
        else:
            class_weights = np.zeros_like(fixed_ann.shape[:2])
            for class_id, class_w in self.wc.items():
                class_weights[fixed_ann == class_id] = class_w
            wmap += class_weights

        # fix other maps to align
        img[fixed_ann == 0] = 0
        img = np.dstack([img, wmap])

        return img


####
class GenInstanceContourMap(GenInstance):
    """
    Input annotation must be of original shape.

    Perform following operation:
        1) Dilate each instance by a kernel with
           a diameter of 7 pix.
        2) Erode each instance by a kernel with
           a diameter of 7 pix.
        3) Obtain the contour by subtracting the
           eroded instance from the dilated instance.

    """

    def __init__(self, crop_shape=None):
        super(GenInstanceContourMap, self).__init__()
        self.crop_shape = crop_shape

    def _augment(self, img, _):
        img = np.copy(img)
        orig_ann = img[..., 0]  # instance ID map
        fixed_ann = self._fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map
        crop_ann = cropping_center(fixed_ann, self.crop_shape)

        # setting 1 boundary pix of each instance to background
        contour_map = np.zeros(fixed_ann.shape[:2], np.uint8)

        inst_list = list(np.unique(crop_ann))
        inst_list.remove(0)  # 0 is background

        k_disk = np.array(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            np.uint8,
        )

        for inst_id in inst_list:
            inst_map = np.array(fixed_ann == inst_id, np.uint8)
            inner = cv2.erode(inst_map, k_disk, iterations=1)
            outer = cv2.dilate(inst_map, k_disk, iterations=1)
            contour_map += outer - inner
        contour_map[contour_map > 0] = 1  # binarize
        img = np.dstack([fixed_ann, contour_map])
        return img


####
class GenInstanceHV(GenInstance):
    """
        Input annotation must be of original shape.

        The map is calculated only for instances within the crop portion
        but based on the original shape in original image.

        Perform following operation:
        Obtain the horizontal and vertical distance maps for each
        nuclear instance.
    """

    def _augment(self, img, _):
        img = np.copy(img)
        orig_ann = img[..., 0]  # instance ID map
        fixed_ann = self._fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map
        crop_ann = cropping_center(fixed_ann, self.crop_shape)
        # TODO: deal with 1 label warning
        crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

        x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(crop_ann))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(fixed_ann == inst_id, np.uint8)
            inst_box = bounding_box(inst_map)

            # expand the box by 2px
            inst_box[0] -= 2
            inst_box[2] -= 2
            inst_box[1] += 2
            inst_box[3] += 2

            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(measurements.center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        img = img.astype("float32")
        img = np.dstack([img, x_map, y_map])

        return img


####
class GenInstanceDistance(GenInstance):
    """
    Input annotation must be of original shape.

    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the standard distance map of nuclear pixels to their closest
    boundary.
    Can be interpreted as the inverse distance map of nuclear pixels to
    the centroid.
    """

    def __init__(self, crop_shape=None, inst_norm=True):
        super(GenInstanceDistance, self).__init__()
        self.crop_shape = crop_shape
        self.inst_norm = inst_norm

    def _augment(self, img, _):
        img = np.copy(img)
        orig_ann = img[..., 0]  # instance ID map
        fixed_ann = self._fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map
        crop_ann = cropping_center(fixed_ann, self.crop_shape)

        orig_dst = np.zeros(orig_ann.shape, dtype=np.float32)

        inst_list = list(np.unique(crop_ann))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(fixed_ann == inst_id, np.uint8)
            inst_box = bounding_box(inst_map)

            # expand the box by 2px
            inst_box[0] -= 2
            inst_box[2] -= 2
            inst_box[1] += 2
            inst_box[3] += 2

            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # chessboard distance map generation
            # normalize distance to 0-1
            inst_dst = distance_transform_cdt(inst_map)
            inst_dst = inst_dst.astype("float32")
            if self.inst_norm:
                max_value = np.amax(inst_dst)
                if max_value <= 0:
                    continue  # HACK: temporay patch for divide 0 i.e no nuclei (how?)
                inst_dst = inst_dst / np.amax(inst_dst)

            ####
            dst_map_box = orig_dst[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            dst_map_box[inst_map > 0] = inst_dst[inst_map > 0]

        #
        img = img.astype("float32")
        img = np.dstack([img, orig_dst])

        return img


####
class GaussianBlur(ImageAugmentor):
    """ Gaussian blur the image with random window size"""

    def __init__(self, max_size=3):
        """
        Args:
            max_size (int): max possible Gaussian window size would be 2 * max_size + 1
        """
        super(GaussianBlur, self).__init__()
        self.max_size = max_size

    def _get_augment_params(self, img):
        sx, sy = self.rng.randint(1, self.max_size, size=(2,))
        sx = sx * 2 + 1
        sy = sy * 2 + 1
        return sx, sy

    def _augment(self, img, s):
        return np.reshape(
            cv2.GaussianBlur(
                img, s, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE
            ),
            img.shape,
        )


####
class BinarizeLabel(ImageAugmentor):
    """ Convert labels to binary maps"""

    def __init__(self):
        super(BinarizeLabel, self).__init__()

    def _get_augment_params(self, img):
        return None

    def _augment(self, img, s):
        img = np.copy(img)
        arr = img[..., 0]
        arr[arr > 0] = 1
        return img


####
class MedianBlur(ImageAugmentor):
    """ Median blur the image with random window size"""

    def __init__(self, max_size=3):
        """
        Args:
            max_size (int): max possible window size
                            would be 2 * max_size + 1
        """
        super(MedianBlur, self).__init__()
        self.max_size = max_size

    def _get_augment_params(self, img):
        s = self.rng.randint(1, self.max_size)
        s = s * 2 + 1
        return s

    def _augment(self, img, ksize):
        return cv2.medianBlur(img, ksize)


class eqHistCV(ImageAugmentor):
    def __init__(self,):
        super(eqHistCV, self).__init__()

    def _get_augment_params(self, img):
        return None

    def _augment(self, img, s):
        R, G, B = cv2.split(np.uint8(img))
        return cv2.merge(
            (cv2.equalizeHist(R), cv2.equalizeHist(G), cv2.equalizeHist(B))
        )


class eqRGB2HED(ImageAugmentor):
    def __init__(self,):
        super(eqRGB2HED, self).__init__()

    def _get_augment_params(self, img):
        return None

    def _augment(self, img, s):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = rgb2hed(img)
        image[..., 0] = equalize_hist(image[..., 0])
        image[..., 1] = equalize_hist(image[..., 1])
        image[..., 2] = equalize_hist(image[..., 2])
        return img_as_ubyte(image)


class pipeHEDAugment(ImageAugmentor):
    def __init__(self,):
        super(pipeHEDAugment, self).__init__()

    def _get_augment_params(self, img):
        return None

    def _augment(self, img, s):
        # R, G, B = cv2.split(np.uint8(img))
        # image = rgb2hed(cv2.merge((cv2.equalizeHist(R), cv2.equalizeHist(G), cv2.equalizeHist(B))))
        # image[:,:,0] = equalize_hist(image[:,:,0])
        # image[:,:,1] = equalize_hist(image[:,:,1])
        # image[:,:,2] = equalize_hist(image[:,:,2])
        # return (img_as_ubyte(image))

        # hed_img = rgb2hed(img)
        # hed_img[..., 0] = rescale_intensity(hed_img[..., 0], out_range=(0, 400)).astype(np.int8)
        # hed_img[..., 1] = rescale_intensity(hed_img[..., 1], out_range=(0, 150)).astype(np.int8)
        # hed_img[..., 2] = rescale_intensity(hed_img[..., 2], out_range=(0, 500)).astype(np.int8)
        # hed_img = np.array(hed_img, dtype=np.uint)
        # return img_as_ubyte(hed_img / np.max(hed_img))
        ihc_hed = rgb2hed(img)
        h = rescale_intensity(ihc_hed[..., 0], out_range=(0, 1))
        d = rescale_intensity(ihc_hed[..., 2], out_range=(0, 1))
        return img_as_ubyte(np.dstack((np.zeros_like(h), d, h)))


class linearAugmentation(ImageAugmentor):
    def __init__(self,):
        super(linearAugmentation, self).__init__()

    def _get_augment_params(self, img):
        return None

    def _augment(self, img, s):
        alpha = [0.95, 1.05]
        bias = [-0.01, 0.01]  # -0.05,  0.05
        hed_img = rgb2hed(img)
        for channel in range(3):
            hed_img[..., channel] *= random.choice(np.arange(alpha[0], alpha[1], 0.01))
            hed_img[..., channel] += random.choice(np.arange(bias[0], bias[1], 0.01))
        return img_as_ubyte(hed2rgb(hed_img))
