import tensorflow as tf

from tensorpack import *
from tensorpack.models import BatchNorm, BNReLU, Conv2D, MaxPooling, FixedUnPooling
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary

from .utils import *

import sys

sys.path.append("..")  # adds higher directory to python modules path
try:
    from config import Config
except ImportError:
    assert False, "Fail to import config.py"

####
def upsample2x(name, x):
    """
    Nearest neighbor up-sampling
    """
    return FixedUnPooling(
        name,
        x,
        2,
        unpool_mat=np.ones((2, 2), dtype="float32"),
        data_format="channels_first",
    )


####
def res_blk(name, l, ch, ksize, count, split=1, strides=1, freeze=False):
    ch_in = l.get_shape().as_list()
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope("block" + str(i)):
                x = l if i == 0 else BNReLU("preact", l)
                x = Conv2D("conv1", x, ch[0], ksize[0], activation=BNReLU)
                x = Conv2D(
                    "conv2",
                    x,
                    ch[1],
                    ksize[1],
                    split=split,
                    strides=strides if i == 0 else 1,
                    activation=BNReLU,
                )
                x = Conv2D("conv3", x, ch[2], ksize[2], activation=tf.identity)
                if (strides != 1 or ch_in[1] != ch[2]) and i == 0:
                    l = Conv2D("convshortcut", l, ch[2], 1, strides=strides)
                x = tf.stop_gradient(x) if freeze else x
                l = l + x
        # end of each group need an extra activation
        l = BNReLU("bnlast", l)
    return l


####
def dense_blk(name, l, ch, ksize, count, split=1, padding="valid"):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope("blk/" + str(i)):
                x = BNReLU("preact_bna", l)
                x = Conv2D(
                    "conv1", x, ch[0], ksize[0], padding=padding, activation=BNReLU
                )
                x = Conv2D("conv2", x, ch[1], ksize[1], padding=padding, split=split)
                ##
                if padding == "valid":
                    x_shape = x.get_shape().as_list()
                    l_shape = l.get_shape().as_list()
                    l = crop_op(l, (l_shape[2] - x_shape[2], l_shape[3] - x_shape[3]))

                l = tf.concat([l, x], axis=1)
        l = BNReLU("blk_bna", l)
    return l


####
def encoder(i, freeze):
    """
    Pre-activated ResNet50 Encoder
    """

    d1 = Conv2D("conv0", i, 64, 7, padding="same", strides=1, activation=BNReLU)
    d1 = res_blk("group0", d1, [64, 64, 256], [1, 3, 1], 3, strides=1, freeze=freeze)

    d2 = res_blk("group1", d1, [128, 128, 512], [1, 3, 1], 4, strides=2, freeze=freeze)
    d2 = tf.stop_gradient(d2) if freeze else d2

    d3 = res_blk("group2", d2, [256, 256, 1024], [1, 3, 1], 6, strides=2, freeze=freeze)
    d3 = tf.stop_gradient(d3) if freeze else d3

    d4 = res_blk("group3", d3, [512, 512, 2048], [1, 3, 1], 3, strides=2, freeze=freeze)
    d4 = tf.stop_gradient(d4) if freeze else d4

    d4 = Conv2D("conv_bot", d4, 1024, 1, padding="same")

    return [d1, d2, d3, d4]


####
def decoder(name, i):
    pad = "valid"  # to prevent boundary artifacts
    with tf.variable_scope(name):
        with tf.variable_scope("u3"):
            u3 = upsample2x("rz", i[-1])
            u3_sum = tf.add_n([u3, i[-2]])

            u3 = Conv2D("conva", u3_sum, 256, 3, strides=1, padding=pad)
            u3 = dense_blk("dense", u3, [128, 32], [1, 3], 8, split=4, padding=pad)
            u3 = Conv2D("convf", u3, 512, 1, strides=1)
        ####
        with tf.variable_scope("u2"):
            u2 = upsample2x("rz", u3)
            u2_sum = tf.add_n([u2, i[-3]])

            u2x = Conv2D("conva", u2_sum, 128, 3, strides=1, padding=pad)
            u2 = dense_blk("dense", u2x, [128, 32], [1, 3], 4, split=4, padding=pad)
            u2 = Conv2D("convf", u2, 256, 1, strides=1)
        ####
        with tf.variable_scope("u1"):
            u1 = upsample2x("rz", u2)
            u1_sum = tf.add_n([u1, i[-4]])

            u1 = Conv2D("conva", u1_sum, 64, 3, strides=1, padding="same")

    return [u3, u2x, u1]


####


class Model(ModelDesc, Config):
    def __init__(self, freeze=False):
        super(Model, self).__init__()
        assert tf.test.is_gpu_available()
        self.freeze = freeze
        self.data_format = "NCHW"

    def _get_inputs(self):
        return [
            InputDesc(tf.float32, [None] + list(self.train_input_shape) + [3], "images"),
            InputDesc(
                tf.float32, [None] + list(self.train_mask_shape) + [None], "truemap-coded"
            ),
        ]

    # for node to receive manual info such as learning rate.
    def add_manual_variable(self, name, init_value, summary=True):
        var = tf.get_variable(name, initializer=init_value, trainable=False)
        if summary:
            tf.summary.scalar(name + "-summary", var)
        return

    def _get_optimizer(self):
        with tf.variable_scope("", reuse=True):
            lr = tf.get_variable("learning_rate")
        opt = self.optimizer(learning_rate=lr)
        return opt


####


class Model_NP_HV_OPT(Model):
    def _build_graph(self, inputs):
        images, truemap_coded = inputs
        orig_imgs = images

        true_type = truemap_coded[..., 1]
        true_type = tf.cast(true_type, tf.int32)
        true_type = tf.identity(true_type, name="truemap-type")
        one_type = tf.one_hot(true_type, self.nr_types, axis=-1)
        true_type = tf.expand_dims(true_type, axis=-1)

        true_np = tf.cast(true_type > 0, tf.int32)  # ? sanity this
        true_np = tf.identity(true_np, name="truemap-np")
        one_np = tf.one_hot(tf.squeeze(true_np), 2, axis=-1)

        true_hv = truemap_coded[..., -2:]
        true_hv = tf.identity(true_hv, name="truemap-hv")
        ####
        with argscope(
            Conv2D,
            activation=tf.identity,
            use_bias=False,  # K.he initializer
            W_init=tf.variance_scaling_initializer(scale=2.0, mode="fan_out"),
        ), argscope([Conv2D, BatchNorm], data_format=self.data_format):

            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0

            ####
            d = encoder(i, self.freeze)
            d[0] = crop_op(d[0], (92, 92))
            d[1] = crop_op(d[1], (36, 36))

            ####
            np_feat = decoder("np", d)
            npx = BNReLU("preact_out_np", np_feat[-1])

            hv_feat = decoder("hv", d)
            hv = BNReLU("preact_out_hv", hv_feat[-1])

            tp_feat = decoder("tp", d)
            tp = BNReLU("preact_out_tp", tp_feat[-1])

            # Nuclei Type Pixels (TP)
            logi_class = Conv2D(
                "conv_out_tp",
                tp,
                self.nr_types,
                1,
                use_bias=True,
                activation=tf.identity,
            )
            logi_class = tf.transpose(logi_class, [0, 2, 3, 1])
            soft_class = tf.nn.softmax(logi_class, axis=-1)

            #### Nuclei Pixels (NP)
            logi_np = Conv2D(
                "conv_out_np", npx, 2, 1, use_bias=True, activation=tf.identity
            )
            logi_np = tf.transpose(logi_np, [0, 2, 3, 1])
            soft_np = tf.nn.softmax(logi_np, axis=-1)
            prob_np = tf.identity(soft_np[..., 1], name="predmap-prob-np")
            prob_np = tf.expand_dims(prob_np, axis=-1)

            #### Horizontal-Vertival (HV)
            logi_hv = Conv2D(
                "conv_out_hv", hv, 2, 1, use_bias=True, activation=tf.identity
            )
            logi_hv = tf.transpose(logi_hv, [0, 2, 3, 1])
            prob_hv = tf.identity(logi_hv, name="predmap-prob-hv")
            pred_hv = tf.identity(logi_hv, name="predmap-hv")

            # * channel ordering: type-map, segmentation map
            # encoded so that inference can extract all output at once
            predmap_coded = tf.concat(
                [soft_class, prob_np, pred_hv], axis=-1, name="predmap-coded"
            )

        # return

        def get_gradient_hv(l, h_ch, v_ch):
            """
            Calculate the horizontal partial differentiation for horizontal channel
            and the vertical partial differentiation for vertical channel.
            The partial differentiation is approximated by calculating the central differnce
            which is obtained by using Sobel kernel of size 5x5. The boundary is zero-padded
            when channel is convolved with the Sobel kernel.
            Args:
                l (tensor): tensor of shape NHWC with C should be 2 (1 channel for horizonal 
                            and 1 channel for vertical)
                h_ch(int) : index within C axis of `l` that corresponds to horizontal channel
                v_ch(int) : index within C axis of `l` that corresponds to vertical channel
            """

            def get_sobel_kernel(size):
                assert size % 2 == 1, "Must be odd, get size=%d" % size

                h_range = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
                v_range = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
                h, v = np.meshgrid(h_range, v_range)
                kernel_h = h / (h * h + v * v + 1.0e-15)
                kernel_v = v / (h * h + v * v + 1.0e-15)
                return kernel_h, kernel_v

            mh, mv = get_sobel_kernel(5)
            mh = tf.constant(mh, dtype=tf.float32)
            mv = tf.constant(mv, dtype=tf.float32)

            mh = tf.reshape(mh, [5, 5, 1, 1])
            mv = tf.reshape(mv, [5, 5, 1, 1])

            # central difference to get gradient, ignore the boundary problem
            h = tf.expand_dims(l[..., h_ch], axis=-1)
            v = tf.expand_dims(l[..., v_ch], axis=-1)
            dh = tf.nn.conv2d(h, mh, strides=[1, 1, 1, 1], padding="SAME")
            dv = tf.nn.conv2d(v, mv, strides=[1, 1, 1, 1], padding="SAME")
            output = tf.concat([dh, dv], axis=-1)
            return output

        def loss_mse(true, pred, name=None):
            ### regression loss
            loss = pred - true
            loss = tf.reduce_mean(loss * loss, name=name)
            return loss

        def loss_msge(true, pred, focus, name=None):
            focus = tf.stack([focus, focus], axis=-1)
            pred_grad = get_gradient_hv(pred, 1, 0)
            true_grad = get_gradient_hv(true, 1, 0)
            loss = pred_grad - true_grad
            loss = focus * (loss * loss)
            # artificial reduce_mean with focus region
            loss = tf.reduce_sum(loss) / (tf.reduce_sum(focus) + 1.0e-8)
            loss = tf.identity(loss, name=name)
            return loss

        ####
        if get_current_tower_context().is_training:
            # ---- LOSS ----#
            loss = 0
            for term, weight in self.loss_term.items():
                if term == "mse":
                    term_loss = loss_mse(true_hv, pred_hv, name="loss-mse")
                elif term == "msge":
                    focus = truemap_coded[..., 0]
                    term_loss = loss_msge(true_hv, pred_hv, focus, name="loss-msge")
                elif term == "bce":
                    term_loss = categorical_crossentropy(soft_np, one_np)
                    term_loss = tf.reduce_mean(term_loss, name="loss-bce")
                elif "dice" in self.loss_term:
                    term_loss = dice_loss(soft_np[..., 0], one_np[..., 0]) + dice_loss(
                        soft_np[..., 1], one_np[..., 1]
                    )
                    term_loss = tf.identity(term_loss, name="loss-dice")
                else:
                    assert False, "Not support loss term: %s" % term
                add_moving_summary(term_loss)
                loss += term_loss * weight

            term_loss = categorical_crossentropy(soft_class, one_type)
            term_loss = tf.reduce_mean(term_loss, name="loss-xentropy-class")
            add_moving_summary(term_loss)
            loss = loss + term_loss
            # term_loss = dice_loss(soft_class[...,0], one_type[...,0]) \

            term_loss = 0
            for type_id in range(self.nr_types):
                term_loss += dice_loss(
                    soft_class[..., type_id], one_type[..., type_id]
                )

            term_loss = tf.identity(term_loss, name="loss-dice-class")
            add_moving_summary(term_loss)
            loss = loss + term_loss

            ### combine the loss into single cost function
            self.cost = tf.identity(loss, name="overall-loss")
            add_moving_summary(self.cost)
            ####

            add_param_summary((".*/W", ["histogram"]))  # monitor W

            ### logging visual sthg
            orig_imgs = tf.cast(orig_imgs, tf.uint8)
            tf.summary.image("input", orig_imgs, max_outputs=1)

            orig_imgs = crop_op(orig_imgs, (92, 92), "NHWC")

            pred_np = colorize(prob_np[..., 0], cmap="jet")
            true_np = colorize(true_np[..., 0], cmap="jet")

            pred_h = colorize(prob_hv[..., 0], vmin=-1, vmax=1, cmap="jet")
            pred_v = colorize(prob_hv[..., 1], vmin=-1, vmax=1, cmap="jet")
            true_h = colorize(true_hv[..., 0], vmin=-1, vmax=1, cmap="jet")
            true_v = colorize(true_hv[..., 1], vmin=-1, vmax=1, cmap="jet")

            pred_type = tf.transpose(soft_class, (0, 1, 3, 2))
            pred_type = tf.reshape(pred_type, [-1, 164, 164 * self.nr_types])
            true_type = tf.cast(true_type[..., 0] / self.nr_classes, tf.float32)
            true_type = colorize(true_type, vmin=0, vmax=1, cmap="jet")
            pred_type = colorize(pred_type, vmin=0, vmax=1, cmap="jet")

            viz = tf.concat(
                [
                    orig_imgs,
                    pred_h,
                    pred_v,
                    pred_np,
                    pred_type,
                    true_h,
                    true_v,
                    true_np,
                    true_type,
                ],
                2,
            )

            viz = tf.concat([viz[0], viz[-1]], axis=0)
            viz = tf.expand_dims(viz, axis=0)
            tf.summary.image("output", viz, max_outputs=1)

        return


# Consolidated hover architecture
import tensorflow as tf

from tensorpack import *
from tensorpack.models import BatchNorm, BNReLU, Conv2D, MaxPooling, FixedUnPooling
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary

from .utils import *

import sys

sys.path.append("..")  # adds higher directory to python modules path.
try:  # HACK: import beyond current level, may need to restructure
    from config import Config
except ImportError:
    assert False, "Fail to import config.py"

####
def upsample2x(name, x):
    """
    Nearest neighbor up-sampling
    """
    return FixedUnPooling(
        name,
        x,
        2,
        unpool_mat=np.ones((2, 2), dtype="float32"),
        data_format="channels_first",
    )


####
def res_blk(name, l, ch, ksize, count, split=1, strides=1, freeze=False):
    ch_in = l.get_shape().as_list()
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope("block" + str(i)):
                x = l if i == 0 else BNReLU("preact", l)
                x = Conv2D("conv1", x, ch[0], ksize[0], activation=BNReLU)
                x = Conv2D(
                    "conv2",
                    x,
                    ch[1],
                    ksize[1],
                    split=split,
                    strides=strides if i == 0 else 1,
                    activation=BNReLU,
                )
                x = Conv2D("conv3", x, ch[2], ksize[2], activation=tf.identity)
                if (strides != 1 or ch_in[1] != ch[2]) and i == 0:
                    l = Conv2D("convshortcut", l, ch[2], 1, strides=strides)
                x = tf.stop_gradient(x) if freeze else x
                l = l + x
        # end of each group need an extra activation
        l = BNReLU("bnlast", l)
    return l


####
def dense_blk(name, l, ch, ksize, count, split=1, padding="valid"):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope("blk/" + str(i)):
                x = BNReLU("preact_bna", l)
                x = Conv2D(
                    "conv1", x, ch[0], ksize[0], padding=padding, activation=BNReLU
                )
                x = Conv2D("conv2", x, ch[1], ksize[1], padding=padding, split=split)
                ##
                if padding == "valid":
                    x_shape = x.get_shape().as_list()
                    l_shape = l.get_shape().as_list()
                    l = crop_op(l, (l_shape[2] - x_shape[2], l_shape[3] - x_shape[3]))

                l = tf.concat([l, x], axis=1)
        l = BNReLU("blk_bna", l)
    return l


####
def encoder(i, freeze):
    """
    Pre-activated ResNet50 Encoder
    """

    d1 = Conv2D("conv0", i, 64, 7, padding="valid", strides=1, activation=BNReLU)
    d1 = res_blk("group0", d1, [64, 64, 256], [1, 3, 1], 3, strides=1, freeze=freeze)

    d2 = res_blk("group1", d1, [128, 128, 512], [1, 3, 1], 4, strides=2, freeze=freeze)
    d2 = tf.stop_gradient(d2) if freeze else d2

    d3 = res_blk("group2", d2, [256, 256, 1024], [1, 3, 1], 6, strides=2, freeze=freeze)
    d3 = tf.stop_gradient(d3) if freeze else d3

    d4 = res_blk("group3", d3, [512, 512, 2048], [1, 3, 1], 3, strides=2, freeze=freeze)
    d4 = tf.stop_gradient(d4) if freeze else d4

    d4 = Conv2D("conv_bot", d4, 1024, 1, padding="same")
    return [d1, d2, d3, d4]


####
def decoder(name, i):
    pad = "valid"  # to prevent boundary artifacts
    with tf.variable_scope(name):
        with tf.variable_scope("u3"):
            u3 = upsample2x("rz", i[-1])
            u3_sum = tf.add_n([u3, i[-2]])

            u3 = Conv2D("conva", u3_sum, 256, 5, strides=1, padding=pad)
            u3 = dense_blk("dense", u3, [128, 32], [1, 5], 8, split=4, padding=pad)
            u3 = Conv2D("convf", u3, 512, 1, strides=1)
        ####
        with tf.variable_scope("u2"):
            u2 = upsample2x("rz", u3)
            u2_sum = tf.add_n([u2, i[-3]])

            u2x = Conv2D("conva", u2_sum, 128, 5, strides=1, padding=pad)
            u2 = dense_blk("dense", u2x, [128, 32], [1, 5], 4, split=4, padding=pad)
            u2 = Conv2D("convf", u2, 256, 1, strides=1)
        ####
        with tf.variable_scope("u1"):
            u1 = upsample2x("rz", u2)
            u1_sum = tf.add_n([u1, i[-4]])

            u1 = Conv2D("conva", u1_sum, 64, 5, strides=1, padding="same")

    return [u3, u2x, u1]


####
class Model(ModelDesc, Config):
    def __init__(self, freeze=False):
        super(Model, self).__init__()
        assert tf.test.is_gpu_available()
        self.freeze = freeze
        self.data_format = "NCHW"

    def _get_inputs(self):
        return [
            InputDesc(tf.float32, [None] + self.train_input_shape + [3], "images"),
            InputDesc(
                tf.float32, [None] + self.train_mask_shape + [None], "truemap-coded"
            ),
        ]

    # for node to receive manual info such as learning rate.
    def add_manual_variable(self, name, init_value, summary=True):
        var = tf.get_variable(name, initializer=init_value, trainable=False)
        if summary:
            tf.summary.scalar(name + "-summary", var)
        return

    def _get_optimizer(self):
        with tf.variable_scope("", reuse=True):
            lr = tf.get_variable("learning_rate")
        opt = self.optimizer(learning_rate=lr)
        return opt


####
class Model_NP_HV(Model):
    def _build_graph(self, inputs):

        images, truemap_coded = inputs
        orig_imgs = images

        true_type = truemap_coded[..., 1]
        true_type = tf.cast(true_type, tf.int32)
        true_type = tf.identity(true_type, name="truemap-type")
        one_type = tf.one_hot(true_type, self.nr_types, axis=-1)
        true_type = tf.expand_dims(true_type, axis=-1)

        true_np = tf.cast(true_type > 0, tf.int32)  # ? sanity this
        true_np = tf.identity(true_np, name="truemap-np")
        one_np = tf.one_hot(tf.squeeze(true_np), 2, axis=-1)

        true_hv = truemap_coded[..., -2:]
        true_hv = tf.identity(true_hv, name="truemap-hv")

        ####
        with argscope(
            Conv2D,
            activation=tf.identity,
            use_bias=False,  # K.he initializer
            W_init=tf.variance_scaling_initializer(scale=2.0, mode="fan_out"),
        ), argscope([Conv2D, BatchNorm], data_format=self.data_format):

            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0

            ####
            d = encoder(i, self.freeze)
            d[0] = crop_op(d[0], (184, 184))
            d[1] = crop_op(d[1], (72, 72))

            ####
            np_feat = decoder("np", d)
            npx = BNReLU("preact_out_np", np_feat[-1])

            hv_feat = decoder("hv", d)
            hv = BNReLU("preact_out_hv", hv_feat[-1])

            tp_feat = decoder("tp", d)
            tp = BNReLU("preact_out_tp", tp_feat[-1])

            # Nuclei Type Pixels (TP)
            logi_class = Conv2D(
                "conv_out_tp",
                tp,
                self.nr_types,
                1,
                use_bias=True,
                activation=tf.identity,
            )
            logi_class = tf.transpose(logi_class, [0, 2, 3, 1])
            soft_class = tf.nn.softmax(logi_class, axis=-1)

            #### Nuclei Pixels (NP)
            logi_np = Conv2D(
                "conv_out_np", npx, 2, 1, use_bias=True, activation=tf.identity
            )
            logi_np = tf.transpose(logi_np, [0, 2, 3, 1])
            soft_np = tf.nn.softmax(logi_np, axis=-1)
            prob_np = tf.identity(soft_np[..., 1], name="predmap-prob-np")
            prob_np = tf.expand_dims(prob_np, axis=-1)

            #### Horizontal-Vertival (HV)
            logi_hv = Conv2D(
                "conv_out_hv", hv, 2, 1, use_bias=True, activation=tf.identity
            )
            logi_hv = tf.transpose(logi_hv, [0, 2, 3, 1])
            prob_hv = tf.identity(logi_hv, name="predmap-prob-hv")
            pred_hv = tf.identity(logi_hv, name="predmap-hv")

            # * channel ordering: type-map, segmentation map
            # encoded so that inference can extract all output at once
            predmap_coded = tf.concat(
                [soft_class, prob_np, pred_hv], axis=-1, name="predmap-coded"
            )
        ####
        def get_gradient_hv(l, h_ch, v_ch):
            """
            Calculate the horizontal partial differentiation for horizontal channel
            and the vertical partial differentiation for vertical channel.
            The partial differentiation is approximated by calculating the central differnce
            which is obtained by using Sobel kernel of size 5x5. The boundary is zero-padded
            when channel is convolved with the Sobel kernel.
            Args:
                l (tensor): tensor of shape NHWC with C should be 2 (1 channel for horizonal 
                            and 1 channel for vertical)
                h_ch(int) : index within C axis of `l` that corresponds to horizontal channel
                v_ch(int) : index within C axis of `l` that corresponds to vertical channel
            """

            def get_sobel_kernel(size):
                assert size % 2 == 1, "Must be odd, get size=%d" % size

                h_range = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
                v_range = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
                h, v = np.meshgrid(h_range, v_range)
                kernel_h = h / (h * h + v * v + 1.0e-15)
                kernel_v = v / (h * h + v * v + 1.0e-15)
                return kernel_h, kernel_v

            mh, mv = get_sobel_kernel(5)
            mh = tf.constant(mh, dtype=tf.float32)
            mv = tf.constant(mv, dtype=tf.float32)

            mh = tf.reshape(mh, [5, 5, 1, 1])
            mv = tf.reshape(mv, [5, 5, 1, 1])

            # central difference to get gradient, ignore the boundary problem
            h = tf.expand_dims(l[..., h_ch], axis=-1)
            v = tf.expand_dims(l[..., v_ch], axis=-1)
            dh = tf.nn.conv2d(h, mh, strides=[1, 1, 1, 1], padding="SAME")
            dv = tf.nn.conv2d(v, mv, strides=[1, 1, 1, 1], padding="SAME")
            output = tf.concat([dh, dv], axis=-1)
            return output

        def loss_mse(true, pred, name=None):
            ### regression loss
            loss = pred - true
            loss = tf.reduce_mean(loss * loss, name=name)
            return loss

        def loss_msge(true, pred, focus, name=None):
            focus = tf.stack([focus, focus], axis=-1)
            pred_grad = get_gradient_hv(pred, 1, 0)
            true_grad = get_gradient_hv(true, 1, 0)
            loss = pred_grad - true_grad
            loss = focus * (loss * loss)
            # artificial reduce_mean with focus region
            loss = tf.reduce_sum(loss) / (tf.reduce_sum(focus) + 1.0e-8)
            loss = tf.identity(loss, name=name)
            return loss

        ####
        if get_current_tower_context().is_training:
            # ---- LOSS ----#
            loss = 0
            for term, weight in self.loss_term.items():
                if term == "mse":
                    term_loss = loss_mse(true_hv, pred_hv, name="loss-mse")
                elif term == "msge":
                    focus = truemap_coded[..., 0]
                    term_loss = loss_msge(true_hv, pred_hv, focus, name="loss-msge")
                elif term == "bce":
                    term_loss = categorical_crossentropy(soft_np, one_np)
                    term_loss = tf.reduce_mean(term_loss, name="loss-bce")
                elif "dice" in self.loss_term:
                    term_loss = dice_loss(soft_np[..., 0], one_np[..., 0]) + dice_loss(
                        soft_np[..., 1], one_np[..., 1]
                    )
                    term_loss = tf.identity(term_loss, name="loss-dice")
                else:
                    assert False, "Not support loss term: %s" % term
                add_moving_summary(term_loss)
                loss += term_loss * weight

            term_loss = categorical_crossentropy(soft_class, one_type)
            term_loss = tf.reduce_mean(term_loss, name="loss-xentropy-class")
            add_moving_summary(term_loss)
            loss = loss + term_loss

            # term_loss = dice_loss(soft_class[...,0], one_type[...,0]) \
            #           + dice_loss(soft_class[...,1], one_type[...,1]) \
            #           + dice_loss(soft_class[...,2], one_type[...,2]) \
            #           + dice_loss(soft_class[...,3], one_type[...,3]) \
            #           + dice_loss(soft_class[...,4], one_type[...,4])

            term_loss = 0
            for type_id in range(self.nr_types):
                term_loss += dice_loss(
                    soft_class[..., type_id], one_type[..., type_id]
                )

            term_loss = tf.identity(term_loss, name="loss-dice-class")
            add_moving_summary(term_loss)
            loss = loss + term_loss

            ### combine the loss into single cost function
            self.cost = tf.identity(loss, name="overall-loss")
            add_moving_summary(self.cost)
            ####

            add_param_summary((".*/W", ["histogram"]))  # monitor W

            ### logging visual sthg
            orig_imgs = tf.cast(orig_imgs, tf.uint8)
            tf.summary.image("input", orig_imgs, max_outputs=1)

            orig_imgs = crop_op(orig_imgs, (190, 190), "NHWC")

            pred_np = colorize(prob_np[..., 0], cmap="jet")
            true_np = colorize(true_np[..., 0], cmap="jet")

            pred_h = colorize(prob_hv[..., 0], vmin=-1, vmax=1, cmap="jet")
            pred_v = colorize(prob_hv[..., 1], vmin=-1, vmax=1, cmap="jet")
            true_h = colorize(true_hv[..., 0], vmin=-1, vmax=1, cmap="jet")
            true_v = colorize(true_hv[..., 1], vmin=-1, vmax=1, cmap="jet")


            pred_type = tf.transpose(soft_class, (0, 1, 3, 2))
            pred_type = tf.reshape(pred_type, [-1, 80, 80 * self.nr_types])
            true_type = tf.cast(true_type[..., 0] / self.nr_classes, tf.float32)
            true_type = colorize(true_type, vmin=0, vmax=1, cmap="jet")
            pred_type = colorize(pred_type, vmin=0, vmax=1, cmap="jet")

            viz = tf.concat(
                [
                    orig_imgs,
                    pred_h,
                    pred_v,
                    pred_np,
                    pred_type,
                    true_h,
                    true_v,
                    true_np,
                    true_type,
                ],
                2,
            )

            viz = tf.concat([viz[0], viz[-1]], axis=0)
            viz = tf.expand_dims(viz, axis=0)
            tf.summary.image("output", viz, max_outputs=1)

        return


####

# Consolidated model utils
import math
import numpy as np

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from matplotlib import cm

# TODO: assert for data format
####
def resize_op(
    x,
    height_factor=None,
    width_factor=None,
    size=None,
    interp="bicubic",
    data_format="channels_last",
):
    """
    Resize by a factor if `size=None` else resize to `size`
    """
    original_shape = x.get_shape().as_list()
    if size is not None:
        if data_format == "channels_first":
            x = tf.transpose(x, [0, 2, 3, 1])
            if interp == "bicubic":
                x = tf.image.resize_bicubic(x, size)
            elif interp == "bilinear":
                x = tf.image.resize_bilinear(x, size)
            else:
                x = tf.image.resize_nearest_neighbor(x, size)
            x = tf.transpose(x, [0, 3, 1, 2])
            x.set_shape(
                (
                    None,
                    original_shape[1] if original_shape[1] is not None else None,
                    size[0],
                    size[1],
                )
            )
        else:
            if interp == "bicubic":
                x = tf.image.resize_bicubic(x, size)
            elif interp == "bilinear":
                x = tf.image.resize_bilinear(x, size)
            else:
                x = tf.image.resize_nearest_neighbor(x, size)
            x.set_shape(
                (
                    None,
                    size[0],
                    size[1],
                    original_shape[3] if original_shape[3] is not None else None,
                )
            )
    else:
        if data_format == "channels_first":
            new_shape = tf.cast(tf.shape(x)[2:], tf.float32)
            new_shape *= tf.constant(
                np.array([height_factor, width_factor]).astype("float32")
            )
            new_shape = tf.cast(new_shape, tf.int32)
            x = tf.transpose(x, [0, 2, 3, 1])
            if interp == "bicubic":
                x = tf.image.resize_bicubic(x, new_shape)
            elif interp == "bilinear":
                x = tf.image.resize_bilinear(x, new_shape)
            else:
                x = tf.image.resize_nearest_neighbor(x, new_shape)
            x = tf.transpose(x, [0, 3, 1, 2])
            x.set_shape(
                (
                    None,
                    original_shape[1] if original_shape[1] is not None else None,
                    int(original_shape[2] * height_factor)
                    if original_shape[2] is not None
                    else None,
                    int(original_shape[3] * width_factor)
                    if original_shape[3] is not None
                    else None,
                )
            )
        else:
            original_shape = x.get_shape().as_list()
            new_shape = tf.cast(tf.shape(x)[1:3], tf.float32)
            new_shape *= tf.constant(
                np.array([height_factor, width_factor]).astype("float32")
            )
            new_shape = tf.cast(new_shape, tf.int32)
            if interp == "bicubic":
                x = tf.image.resize_bicubic(x, new_shape)
            elif interp == "bilinear":
                x = tf.image.resize_bilinear(x, new_shape)
            else:
                x = tf.image.resize_nearest_neighbor(x, new_shape)
            x.set_shape(
                (
                    None,
                    int(original_shape[1] * height_factor)
                    if original_shape[1] is not None
                    else None,
                    int(original_shape[2] * width_factor)
                    if original_shape[2] is not None
                    else None,
                    original_shape[3] if original_shape[3] is not None else None,
                )
            )
    return x


####
def crop_op(x, cropping, data_format="channels_first"):
    """
    Center crop image
    Args:
        cropping is the substracted portion
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "channels_first":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r]
    return x


####


def categorical_crossentropy(output, target):
    """
        categorical cross-entropy, accept probabilities not logit
    """
    # scale preds so that the class probs of each sample sum to 1
    output /= tf.reduce_sum(
        output, reduction_indices=len(output.get_shape()) - 1, keepdims=True
    )
    # manual computation of crossentropy
    epsilon = tf.convert_to_tensor(10e-8, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1.0 - epsilon)
    return -tf.reduce_sum(
        target * tf.log(output), reduction_indices=len(output.get_shape()) - 1
    )


####
def dice_loss(output, target, loss_type="sorensen", axis=None, smooth=1e-3):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), 
              dice = ```smooth/(small_value + smooth)``, then if smooth is very small, 
              dice close to 0 (even the image values lower than the threshold), 
              so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> dice_loss = dice_coe(outputs, y_)
    """
    target = tf.squeeze(tf.cast(target, tf.float32))
    output = tf.squeeze(tf.cast(output, tf.float32))

    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == "jaccard":
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == "sorensen":
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknown loss_type")
    # already flatten
    dice = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    ##
    return dice


####
def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    Arguments:
      - value: input tensor, NHWC ('channels_last')
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3], uint8.
    """

    # normalize
    if vmin is None:
        vmin = tf.reduce_min(value, axis=[1, 2])
        vmin = tf.reshape(vmin, [-1, 1, 1])
    if vmax is None:
        vmax = tf.reduce_max(value, axis=[1, 2])
        vmax = tf.reshape(vmax, [-1, 1, 1])
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    # NOTE: will throw error if use get_shape()
    # value = tf.squeeze(value)

    # quantize
    value = tf.round(value * 255)
    indices = tf.cast(value, np.int32)

    # gather
    colormap = cm.get_cmap(cmap if cmap is not None else "gray")
    colors = colormap(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
    value = tf.cast(value * 255, tf.uint8)
    return value


####
def make_image(x, cy, cx, scale_y, scale_x):
    """
    Take 1st image from x and turn channels representations
    into 2D image, with cx number of channels in x-axis and
    cy number of channels in y-axis
    """
    # norm x for better visual
    x = tf.transpose(x, (0, 2, 3, 1))  # NHWC
    max_x = tf.reduce_max(x, axis=-1, keep_dims=True)
    min_x = tf.reduce_min(x, axis=-1, keep_dims=True)
    x = 255 * (x - min_x) / (max_x - min_x)
    ###
    x_shape = tf.shape(x)
    channels = x_shape[-1]
    iy, ix = x_shape[1], x_shape[2]
    ###
    x = tf.slice(x, (0, 0, 0, 0), (1, -1, -1, -1))
    x = tf.reshape(x, (iy, ix, channels))
    ix += 4
    iy += 4
    x = tf.image.resize_image_with_crop_or_pad(x, iy, ix)
    x = tf.reshape(x, (iy, ix, cy, cx))
    x = tf.transpose(x, (2, 0, 3, 1))  # cy,iy,cx,ix
    x = tf.reshape(x, (1, cy * iy, cx * ix, 1))
    x = resize_op(x, scale_y, scale_x)
    return tf.cast(x, tf.uint8)


####
