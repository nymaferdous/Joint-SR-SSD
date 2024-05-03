#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
import os
import shutil
from os.path import isfile, isdir
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
from tensorflow.python import pywrap_tensorflow
import math
import sys

import pickle
import cv2
import os

import multiprocessing as mp
import numpy as np
import queue as q
import tensorflow as tf

from data_queue import DataQueue
from copy import copy
from utils import *
from ssdutils import *
from transforms import *

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g_custom, SRGAN_d, Vgg19_simple_api
from utils1 import *
from config import config, log_config
import multiprocessing as mp

from training_data import TrainingData
from ssdutils import *
from ssdvgg import *
from utils import *
from average_precision import APCalculator, APs2mAP


## Detector_model
import random
from tqdm import tqdm
import cv2



###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))

in_size = 128

###======================Detector=======================###


def compute_lr(lr_values, lr_boundaries):
    with tf.variable_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)
    return lr, global_step


def __with_loss(x, l2_loss):
    l2_loss += l2_loss
    return x

def build_from_vgg(vgg_dir, num_classes, sess,preset,a_trous=True,
                   progress_hook='tqdm'):
    """
    Build the model for training based on a pre-define vgg16 model.
    :param vgg_dir:       directory where the vgg model should be stored
    :param num_classes:   number of classes
    :param progress_hook: a hook to show download progress of vgg16;
                          the value may be a callable for urlretrieve
                          or string "tqdm"
    """

    num_classes = num_classes+1
    num_vars = num_classes+5
    l2_loss = 0
    # __download_vgg(vgg_dir, progress_hook)
    sess = sess
    preset=preset

    graph = tf.saved_model.loader.load(sess, ['vgg16'], vgg_dir + '/vgg')
    with tf.variable_scope('ssd'):
        image_input = sess.graph.get_tensor_by_name('image_input:0')
        # self.image_input = self.image_input
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        vgg_conv3_3 = sess.graph.get_tensor_by_name('conv3_3/Relu:0')
        vgg_conv4_3 = sess.graph.get_tensor_by_name('conv4_3/Relu:0')
        vgg_conv5_3 = sess.graph.get_tensor_by_name('conv5_3/Relu:0')
        vgg_fc6_w = sess.graph.get_tensor_by_name('fc6/weights:0')
        vgg_fc6_b = sess.graph.get_tensor_by_name('fc6/biases:0')
        vgg_fc7_w = sess.graph.get_tensor_by_name('fc7/weights:0')
        vgg_fc7_b = sess.graph.get_tensor_by_name('fc7/biases:0')

        layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                  'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
                  'conv5_1', 'conv5_2', 'conv5_3']

        for l in layers:
            l2_loss += sess.graph.get_tensor_by_name(l + '/L2Loss:0')


            mod_pool5 = tf.nn.max_pool(vgg_conv5_3, ksize=[1, 3, 3, 1],
                                            strides=[1, 1, 1, 1], padding='SAME',
                                            name='mod_pool5')

        with tf.variable_scope('mod_conv6'):
            x = tf.nn.conv2d(mod_pool5, vgg_fc6_w,
                             strides=[1, 1, 1, 1], padding='SAME')
            # x = tf.nn.conv2d(self.vgg_conv5_3, self.vgg_fc6_w,
            #                         strides=[1, 2, 2, 1], padding='SAME')
            x = tf.nn.bias_add(x, vgg_fc6_b)
            mod_conv6 = tf.nn.relu(x)
            l2_loss += tf.nn.l2_loss(vgg_fc6_w)

        with tf.variable_scope('mod_conv7'):
            x = tf.nn.conv2d(mod_conv6, vgg_fc7_w,
                             strides=[1, 1, 1, 1], padding='SAME')
            x = tf.nn.bias_add(x, vgg_fc7_b)
            x = tf.nn.relu(x)
            mod_conv7 = x
            l2_loss += tf.nn.l2_loss(vgg_fc7_w)
    # if a_trous: self.__build_vgg_mods_a_trous()
    # else:

        x = l2_normalization(vgg_conv3_3, 20, 256, 'l2_norm_conv3_3')
        norm_conv3_3 = x
        x = l2_normalization(vgg_conv4_3, 20, 512, 'l2_norm_conv4_3')
        norm_conv4_3 = x

        x, l2 = conv_map(mod_conv7, 256, 1, 1, 'ssd_conv8_1')
        ssd_conv8_1 = __with_loss(x, l2)
        x, l2 = conv_map(ssd_conv8_1, 512, 3, 2, 'ssd_conv8_2')
        ssd_conv8_2 = __with_loss(x, l2)
        x, l2 = conv_map(ssd_conv8_2, 128, 1, 1, 'ssd_conv9_1')
        ssd_conv9_1 = __with_loss(x, l2)
        x, l2 = conv_map(ssd_conv9_1, 256, 3, 2, 'ssd_conv9_2')
        ssd_conv9_2 = __with_loss(x, l2)
        x, l2 = conv_map(ssd_conv9_2, 128, 1, 1, 'ssd_conv10_1')
        ssd_conv10_1 = __with_loss(x, l2)
        x, l2 = conv_map(ssd_conv10_1, 256, 3, 2, 'ssd_conv10_2', 'SAME')
        ssd_conv10_2 = __with_loss(x, l2)
        x, l2 = conv_map(ssd_conv10_2, 128, 1, 1, 'ssd_conv11_1')
        ssd_conv11_1 = __with_loss(x, l2)
        x, l2 = conv_map(ssd_conv11_1, 256, 3, 1, 'ssd_conv11_2', 'VALID')
        ssd_conv11_2 = __with_loss(x, l2)

        if len(preset.maps) < 7:
            return

        x, l2 = conv_map(ssd_conv11_2, 128, 1, 1, 'ssd_conv12_1')
        paddings = [[0, 0], [0, 1], [0, 1], [0, 0]]
        x = tf.pad(x, paddings, "CONSTANT")
        ssd_conv12_1 = __with_loss(x, l2)
        x, l2 = conv_map(ssd_conv12_1, 256, 3, 1, 'ssd_conv12_2', 'VALID')
        ssd_conv12_2 = __with_loss(x, l2)

        __maps = [
            norm_conv3_3,
            norm_conv4_3,
            mod_conv7,
            ssd_conv8_2,
            ssd_conv9_2,
            ssd_conv10_2,
            ssd_conv11_2]

        def classifier(x, size, mapsize, name):
            with tf.variable_scope(name):
                w = tf.get_variable("filter",
                                    shape=[3, 3, x.get_shape()[3], size],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.zeros(size), name='biases')
                x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
                x = tf.nn.bias_add(x, b)
                x = tf.reshape(x, [-1, mapsize.w * mapsize.h, size])
                l2 = tf.nn.l2_loss(w)
            return x, l2

        with tf.variable_scope('classifiers', reuse=tf.AUTO_REUSE):
            __classifiers = []
            for i in range(len(__maps)):
                fmap = __maps[i]
                map_size = preset.maps[i].size

                for j in range(2 + len(preset.maps[i].aspect_ratios)):
                    name = 'classifier{}_{}'.format(i, j)
                    clsfier, l2 = classifier(fmap, num_vars, map_size, name)
                    print("Classifier", clsfier.shape)
                    __classifiers.append(__with_loss(clsfier, l2))

        with tf.variable_scope('output'):
            output = tf.concat(__classifiers, axis=1, name='output')
            logits = output[:, :, :num_classes]

        with tf.variable_scope('result'):
            classifier = tf.nn.softmax(logits)
            print("Classifier", classifier)
            locator = output[:, :, num_classes:]
            result = tf.concat([classifier, locator],
                               axis=-1, name='result')


        labels = tf.placeholder(tf.float32, name='labels', shape=[None, None, num_vars])

        with tf.variable_scope('ground_truth'):
            # -------------------------------------------------------------------
            # Split the ground truth tensor
            # -------------------------------------------------------------------
            # Classification ground truth tensor
            # Shape: (batch_size, num_anchors, num_classes)
            gt_cl = labels[:, :, :num_classes]
            print("Number of classes", num_classes)

            # Localization ground truth tensor
            # Shape: (batch_size, num_anchors, 4)
            gt_loc = labels[:, :, num_classes:]

            # Batch size
            # Shape: scalar
            batch_size = tf.shape(gt_cl)[0]

        # -----------------------------------------------------------------------
        # Compute match counters
        # -----------------------------------------------------------------------
        with tf.variable_scope('match_counters'):
            # Number of anchors per sample
            # Shape: (batch_size)
            total_num = tf.ones([batch_size], dtype=tf.int64) * \
                        tf.to_int64(preset.num_anchors)

            # Number of negative (not-matched) anchors per sample, computed
            # by counting boxes of the background class in each sample.
            # Shape: (batch_size)
            negatives_num = tf.count_nonzero(gt_cl[:, :, -1], axis=1)

            # Number of positive (matched) anchors per sample
            # Shape: (batch_size)
            positives_num = total_num - negatives_num

            # Number of positives per sample that is division-safe
            # Shape: (batch_size)
            positives_num_safe = tf.where(tf.equal(positives_num, 0),
                                          tf.ones([batch_size]) * 10e-15,
                                          tf.to_float(positives_num))

        # -----------------------------------------------------------------------
        # Compute masks
        # -----------------------------------------------------------------------
        with tf.variable_scope('match_masks'):
            # Boolean tensor determining whether an anchor is a positive
            # Shape: (batch_size, num_anchors)
            positives_mask = tf.equal(gt_cl[:, :, -1], 0)

            # Boolean tensor determining whether an anchor is a negative
            # Shape: (batch_size, num_anchors)
            negatives_mask = tf.logical_not(positives_mask)

        # -----------------------------------------------------------------------
        # Compute the confidence loss
        # -----------------------------------------------------------------------
        with tf.variable_scope('confidence_loss'):
            # Cross-entropy tensor - all of the values are non-negative
            # Shape: (batch_size, num_anchors)

            ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_cl,
                                                            logits=logits)

            # -------------------------------------------------------------------
            # Sum up the loss of all the positive anchors
            # -------------------------------------------------------------------
            # Positives - the loss of negative anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            positives = tf.where(positives_mask, ce, tf.zeros_like(ce))

            # Total loss of positive anchors
            # Shape: (batch_size)
            positives_sum = tf.reduce_sum(positives, axis=-1)

            # -------------------------------------------------------------------
            # Figure out what the negative anchors with highest confidence loss
            # are
            # -------------------------------------------------------------------
            # Negatives - the loss of positive anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            negatives = tf.where(negatives_mask, ce, tf.zeros_like(ce))

            # Top negatives - sorted confience loss with the highest one first
            # Shape: (batch_size, num_anchors)
            negatives_top = tf.nn.top_k(negatives, preset.num_anchors)[0]

            # -------------------------------------------------------------------
            # Fugure out what the number of negatives we want to keep is
            # -------------------------------------------------------------------
            # Maximum number of negatives to keep per sample - we keep at most
            # 3 times as many as we have positive anchors in the sample
            # Shape: (batch_size)
            negatives_num_max = tf.minimum(negatives_num, 3 * positives_num)

            # -------------------------------------------------------------------
            # Mask out superfluous negatives and compute the sum of the loss
            # -------------------------------------------------------------------
            # Transposed vector of maximum negatives per sample
            # Shape (batch_size, 1)
            negatives_num_max_t = tf.expand_dims(negatives_num_max, 1)

            # Range tensor: [0, 1, 2, ..., num_anchors-1]
            # Shape: (num_anchors)
            rng = tf.range(0, preset.num_anchors, 1)

            # Row range, the same as above, but int64 and a row of a matrix
            # Shape: (1, num_anchors)
            range_row = tf.to_int64(tf.expand_dims(rng, 0))

            # Mask of maximum negatives - first `negative_num_max` elements
            # in corresponding row are `True`, the rest is false
            # Shape: (batch_size, num_anchors)
            negatives_max_mask = tf.less(range_row, negatives_num_max_t)

            # Max negatives - all the positives and superfluous negatives are
            # zeroed out.
            # Shape: (batch_size, num_anchors)
            negatives_max = tf.where(negatives_max_mask, negatives_top,
                                     tf.zeros_like(negatives_top))

            # Sum of max negatives for each sample
            # Shape: (batch_size)
            negatives_max_sum = tf.reduce_sum(negatives_max, axis=-1)

            # -------------------------------------------------------------------
            # Compute the confidence loss for each element
            # -------------------------------------------------------------------
            # Total confidence loss for each sample
            # Shape: (batch_size)
            confidence_loss = tf.add(positives_sum, negatives_max_sum)

            # Total confidence loss normalized by the number of positives
            # per sample
            # Shape: (batch_size)
            confidence_loss = tf.where(tf.equal(positives_num, 0),
                                       tf.zeros([batch_size]),
                                       tf.div(confidence_loss,
                                              positives_num_safe))

            # Mean confidence loss for the batch
            # Shape: scalar

            confidence_loss = tf.reduce_mean(confidence_loss,
                                             name='confidence_loss')

        # -----------------------------------------------------------------------
        # Compute the localization loss
        # -----------------------------------------------------------------------
        with tf.variable_scope('localization_loss'):
            # Element-wise difference between the predicted localization loss
            # and the ground truth
            # Shape: (batch_size, num_anchors, 4)
            loc_diff = tf.subtract(locator, gt_loc)

            # Smooth L1 loss
            # Shape: (batch_size, num_anchors, 4)
            loc_loss = smooth_l1_loss(loc_diff)

            # Sum of localization losses for each anchor
            # Shape: (batch_size, num_anchors)
            loc_loss_sum = tf.reduce_sum(loc_loss, axis=-1)

            # Positive locs - the loss of negative anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            positive_locs = tf.where(positives_mask, loc_loss_sum,
                                     tf.zeros_like(loc_loss_sum))

            # Total loss of positive anchors
            # Shape: (batch_size)
            localization_loss = tf.reduce_sum(positive_locs, axis=-1)

            # Total localization loss normalized by the number of positives
            # per sample
            # Shape: (batch_size)
            localization_loss = tf.where(tf.equal(positives_num, 0),
                                         tf.zeros([batch_size]),
                                         tf.div(localization_loss,
                                                positives_num_safe))

            # Mean localization loss for the batch
            # Shape: scalar
            localization_loss = tf.reduce_mean(localization_loss,
                                               name='localization_loss')

        # -----------------------------------------------------------------------
        # Compute total loss
        # -----------------------------------------------------------------------
        with tf.variable_scope('total_loss'):
            # Sum of the localization and confidence loss
            # Shape: (batch_size)
            conf_and_loc_loss = tf.add(confidence_loss,localization_loss,
                                       name='sum_losses')

            # L2 loss
            # Shape: scalar
            l2_loss = tf.multiply(weight_decay=0.005, l2_loss,
                                  name='l2_loss')

            # Final loss
            # Shape: scalar
            loss = tf.add(conf_and_loc_loss, l2_loss, name='loss')

            losses = {
                'total': loss,
                'localization': localization_loss,
                'confidence': confidence_loss,
                'l2': l2_loss
            }
        return loss

def build_optimizer(learning_rate=0.001, weight_decay=0.005,
                    momentum=0.2, global_step=None,loss):

    #-----------------------------------------------------------------------
    # Build the optimizer
    #-----------------------------------------------------------------------
    with tf.variable_scope('optimizer',reuse=tf.AUTO_REUSE):
        # detector_vars = []
        # with tf.name_scope("define_first_stage_train"):
        #     first_stage_trainable_var_list = []
        #     for var in tf.trainable_variables():
        #         var_name = var.op.name
        #         var_name_mess = str(var_name).split('/')
        #         if var_name_mess[0] in ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
        #                                 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']:
        #             detector_vars.append(var)
        # detector_vars = detector_vars

        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        optimizer = optimizer.minimize(loss, global_step=global_step,
                                       name='optimizer')

    #-----------------------------------------------------------------------
    # Store the tensors
    #-----------------------------------------------------------------------
    optimizer = optimizer


#-------------------------------------------------------------------------------

def train():
    parser = argparse.ArgumentParser(description='Train the SSD')
    parser.add_argument('--name', default='test',
                        help='project name')
    parser.add_argument('--data-dir', default='pascal-voc',
                        help='data directory')
    parser.add_argument('--vgg-dir', default='vgg_graph',
                        help='directory for the VGG-16 model')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--tensorboard-dir', default="tb",
                        help='name of the tensorboard data directory')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='checkpoint interval')
    parser.add_argument('--lr-values', type=str, default='0.001;0.0001;0.00001',
                        help='learning rate values')
    parser.add_argument('--lr-boundaries', type=str, default='320000;400000',
                        help='learning rate chage boundaries (in batches)')
    parser.add_argument('--momentum', type=float, default=0.7,
                        help='momentum for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.008,
                        help='L2 normalization factor')
    parser.add_argument('--continue-training', type=str2bool, default='False',
                        help='continue training from the latest checkpoint')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count(),
                        help='number of parallel generators')

    parser.add_argument('--data-source', default='pascal_voc',
                        help='data source')
    parser.add_argument('--validation-fraction', type=float, default=0.025,
                        help='fraction of the data to be used for validation')
    parser.add_argument('--expand-probability', type=float, default=0.5,
                        help='probability of running sample expander')
    parser.add_argument('--sampler-trials', type=int, default=1,
                        help='number of time a sampler tries to find a sample')
    parser.add_argument('--annotate', type=str2bool, default='False',
                        help="Annotate the data samples")
    parser.add_argument('--compute-td', type=str2bool, default='True',
                        help="Compute training data")
    parser.add_argument('--preset', default='vgg512',
                        choices=['vgg300', 'vgg512'],
                        help="The neural network preset")
    parser.add_argument('--process-test', type=str2bool, default='False',
                        help="process the test dataset")

    args = parser.parse_args()

    print('[i] Project name:         ', args.name)
    print('[i] Data directory:       ', args.data_dir)
    print('[i] VGG directory:        ', args.vgg_dir)
    print('[i] # epochs:             ', args.epochs)
    print('[i] Batch size:           ', args.batch_size)
    print('[i] Tensorboard directory:', args.tensorboard_dir)
    print('[i] Checkpoint interval:  ', args.checkpoint_interval)
    print('[i] Learning rate values: ', args.lr_values)
    print('[i] Learning rate boundaries: ', args.lr_boundaries)
    print('[i] Momentum:             ', args.momentum)
    print('[i] Weight decay:         ', args.weight_decay)
    print('[i] Continue:             ', args.continue_training)
    print('[i] Number of workers:    ', args.num_workers)

    print('[i] Data source:          ', args.data_source)
    print('[i] Validation fraction:  ', args.validation_fraction)
    print('[i] Expand probability:   ', args.expand_probability)
    print('[i] Sampler trials:       ', args.sampler_trials)
    print('[i] Annotate:             ', args.annotate)
    print('[i] Compute training data:', args.compute_td)
    print('[i] Preset:               ', args.preset)
    print('[i] Process test dataset: ', args.process_test)

    # ---------------------------------------------------------------------------
    # Find an existing checkpoint
    # ---------------------------------------------------------------------------

    def build_sampler(overlap, trials):
        return SamplerTransform(sample=True, min_scale=0.3, max_scale=1.0,
                                min_aspect_ratio=0.5, max_aspect_ratio=2.0,
                                min_jaccard_overlap=overlap, max_trials=trials)

    def build_train_transforms(preset, num_classes, sampler_trials, expand_prob):
        # ---------------------------------------------------------------------------
        # Resizing
        # ---------------------------------------------------------------------------
        tf_resize = ResizeTransform(width=preset.image_size.w,
                                    height=preset.image_size.h,
                                    algorithms=[cv2.INTER_LINEAR,
                                                cv2.INTER_AREA,
                                                cv2.INTER_NEAREST,
                                                cv2.INTER_CUBIC,
                                                cv2.INTER_LANCZOS4])
        #
        # #---------------------------------------------------------------------------
        # # Image distortions
        # #---------------------------------------------------------------------------
        # tf_brightness = BrightnessTransform(delta=32)
        # tf_rnd_brightness = RandomTransform(prob=0.5, transform=tf_brightness)
        #
        # tf_contrast = ContrastTransform(lower=0.5, upper=1.5)
        # tf_rnd_contrast = RandomTransform(prob=0.5, transform=tf_contrast)
        #
        # tf_hue = HueTransform(delta=18)
        # tf_rnd_hue = RandomTransform(prob=0.5, transform=tf_hue)
        #
        # tf_saturation = SaturationTransform(lower=0.5, upper=1.5)
        # tf_rnd_saturation = RandomTransform(prob=0.5, transform=tf_saturation)
        #
        tf_reorder_channels = ReorderChannelsTransform()
        tf_rnd_reorder_channels = RandomTransform(prob=0.5,
                                                  transform=tf_reorder_channels)
        #
        # #---------------------------------------------------------------------------
        # # Compositions of image distortions
        # #---------------------------------------------------------------------------
        # tf_distort_lst = [
        #     tf_rnd_contrast,
        #     tf_rnd_saturation,
        #     tf_rnd_hue,
        #     tf_rnd_contrast
        # ]
        # tf_distort_1 = ComposeTransform(transforms=tf_distort_lst[:-1])
        # tf_distort_2 = ComposeTransform(transforms=tf_distort_lst[1:])
        # tf_distort_comp = [tf_distort_1, tf_distort_2]
        # tf_distort = TransformPickerTransform(transforms=tf_distort_comp)
        #
        # #---------------------------------------------------------------------------
        # # Expand sample
        # #---------------------------------------------------------------------------
        tf_expand = ExpandTransform(max_ratio=4.0, mean_value=[104, 117, 123])
        tf_rnd_expand = RandomTransform(prob=expand_prob, transform=tf_expand)
        #
        # #---------------------------------------------------------------------------
        # # Samplers
        # #---------------------------------------------------------------------------
        samplers = [
            SamplerTransform(sample=False),
            build_sampler(0.1, sampler_trials),
            build_sampler(0.3, sampler_trials),
            build_sampler(0.5, sampler_trials),
            build_sampler(0.7, sampler_trials),
            build_sampler(0.9, sampler_trials),
            build_sampler(1.0, sampler_trials)
        ]
        tf_sample_picker = SamplePickerTransform(samplers=samplers)
        #
        # #---------------------------------------------------------------------------
        # # Horizontal flip
        # #---------------------------------------------------------------------------
        # tf_flip = HorizontalFlipTransform()
        # tf_rnd_flip = RandomTransform(prob=0.5, transform=tf_flip)
        #
        # #---------------------------------------------------------------------------
        # Transform list
        # ---------------------------------------------------------------------------
        transforms = [
            ImageLoaderTransform(),
            # tf_rnd_brightness,
            # tf_distort,
            # tf_rnd_reorder_channels,
            # tf_rnd_expand,
            # tf_sample_picker,
            # tf_rnd_flip,
            LabelCreatorTransform(preset=preset, num_classes=num_classes)
            # tf_resize
        ]
        return transforms

    ##From traiining data
    # if isdir(args.name):
    #     shutil.rmtree(args.name)

    source = load_data_source(args.data_source)
    source.load_trainval_data(args.data_dir, args.validation_fraction)
    nones = [None] * len(source.train_samples)
    train_samples = list(zip(nones, nones, source.train_samples))
    nones = [None] * len(source.valid_samples)
    valid_samples = list(zip(nones, nones, source.valid_samples))

    nones = [None] * len(train_samples)
    train_samples = list(zip(nones, nones, train_samples))
    nones = [None] * len(valid_samples)
    valid_samples = list(zip(nones, nones, valid_samples))


    preset = get_preset_by_name(args.preset)
    num_classes = source.num_classes
    label_colors = source.colors
    lid2name = source.lid2name
    lname2id = source.lname2id
    train_tfs = build_train_transforms(preset, num_classes, args.sampler_trials,
                                    args.expand_probability)
    num_train = len(train_samples)
    num_valid = len(valid_samples)
    train_samples = list(map(lambda x: x[2], train_samples))


    def run_transforms(sample):
        args = sample
        print("Run transform", args)

        for t in train_tfs:
            args = t(*args)

        return args

    def process_samples(samples):
        images=[]
        labels = []
        gt_boxes = []
        done = False

        image, label, gt = run_transforms(samples)
        print("Processing Label", label.shape)
        # label, gt = Labelcreatortransform(samples, preset=preset, num_classes=num_classes)
        num_bg = np.count_nonzero(label[:, num_classes])
        done = num_bg < label.shape[0]

        images.append(image.astype(np.float32))
        labels.append(label.astype(np.float32))
        gt_boxes.append(gt.boxes)
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        return images, labels, gt_boxes


    def gen_batch(sample_list_, idx):
        images=[]
        labels = []
        gt_boxes = []
        sample_list = copy(sample_list_[idx])
        print("Sample List",sample_list)
        samples = sample_list
        print("Generated Sample", samples)

        images, labels, gt_boxes = process_samples(samples)
        print("Labels Genearted",labels.shape)
        return images, labels, gt_boxes


    if isdir('checkpoint_detector'):
        shutil.rmtree('checkpoint_detector')
    start_epoch = 0

    if args.continue_training:
        state = tf.train.get_checkpoint_state(args.name)
        if state is None:
            print('[!] No network state found in ' + args.name)
            return 1

        ckpt_paths = state.all_model_checkpoint_paths
        print(ckpt_paths)
        if not ckpt_paths:
            print('[!] No network state found in ' + args.name)
            return 1

        last_epoch = None
        checkpoint_file = None
        for ckpt in ckpt_paths:
            ckpt_num = os.path.basename(ckpt).split('.')[0][1:]
            try:
                ckpt_num = int(ckpt_num)
            except ValueError:
                continue
            if last_epoch is None or last_epoch < ckpt_num:
                last_epoch = ckpt_num
                checkpoint_file = ckpt

        if checkpoint_file is None:
            print('[!] No checkpoints found, cannot continue!')
            return 1

        metagraph_file = checkpoint_file + '.meta'

        if not os.path.exists(metagraph_file):
            print('[!] Cannot find metagraph', metagraph_file)
            return 1
        start_epoch = last_epoch


    # ---------------------------------------------------------------------------
    # Create a project directory
    # ---------------------------------------------------------------------------
    # else:
    #     try:
    #         print('[i] Creating directory {}...'.format(args.name))
    #         os.makedirs(args.name)
    #     except (IOError) as e:
    #         print('[!]', str(e))
    #         return 1

    print('[i] Starting at epoch:    ', start_epoch + 1)

    # Configure the training data
    # ---------------------------------------------------------------------------


    ## create folders to save result images and trained model
    save_dir_ginit = "samples_celeba/{}_ginit_{}".format(tl.global_flag['mode'], in_size)
    save_dir_gan = "samples_celeba/{}_gan_{}".format(tl.global_flag['mode'], in_size)
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint_{}".format(in_size)  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.jpg', printable=False))
    # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.jpg', printable=False))
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.jpg', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, in_size, in_size, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 256, 256, 3], name='t_target_image')
    t_target_image512 = tf.placeholder('float32', [batch_size, 512, 512, 3], name='t_target_image')

    ## Detector_Placeholder

    net_g, net_g512 = SRGAN_g_custom(t_image, is_train=True, reuse=False)

    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    net_d512, logits_real512 = SRGAN_d(t_target_image512, is_train=True, reuse=False, scope="SRGAN_d512")

    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)
    _, logits_fake512 = SRGAN_d(net_g512.outputs, is_train=True, reuse=True, scope="SRGAN_d512")


    ### Detector_model
    # model = YOLOV3(net_g512.outputs, trainable)


    net_g.print_params(False)
    net_g.print_layers()
    net_g512.print_params(False)
    net_g512.print_layers()
    net_d.print_params(False)
    net_d.print_layers()

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False)
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)

    t_target_image_224_512 = tf.image.resize_images(t_target_image512, size=[224, 224], method=0, align_corners=False)
    t_predict_image_224_512 = tf.image.resize_images(net_g512.outputs, size=[224, 224], method=0, align_corners=False)

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    _, vgg_target_emb512 = Vgg19_simple_api((t_target_image_224_512 + 1) / 2, reuse=True)
    _, vgg_predict_emb512 = Vgg19_simple_api((t_predict_image_224_512 + 1) / 2, reuse=True)

    ## test inference
    net_g_test_256, net_g_test_512 = SRGAN_g_custom(t_image, is_train=False, reuse=True)

    ####========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')

    d_loss3 = tl.cost.sigmoid_cross_entropy(logits_real512, tf.ones_like(logits_real512), name='d3')
    d_loss4 = tl.cost.sigmoid_cross_entropy(logits_fake512, tf.zeros_like(logits_fake512), name='d4')

    d_loss = d_loss3 + d_loss4 + d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_gan_loss512 = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake512, tf.ones_like(logits_fake512), name='g2')
    mse_loss512 = tl.cost.mean_squared_error(net_g512.outputs, t_target_image512, is_mean=True)
    vgg_loss512 = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb512.outputs, vgg_target_emb512.outputs, is_mean=True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ##Detector_Loss
    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)
    d_vars512 = tl.layers.get_variables_with_name('SRGAN_d512', True, True)
    d_vars.extend(d_vars512)

    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss + mse_loss512, var_list=g_vars)
    ## SRGAN
    g_loss = mse_loss + vgg_loss + g_gan_loss + mse_loss512 + vgg_loss512 + g_gan_loss512
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)


    # detector_vars = tf.global_variables()
    # print(detector_vars)

    ## SRGAN Try


    ###========================== RESTORE MODEL =============================###
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    with tf.Session() as sess:
        net = SSDVGG(sess, preset)
        # tf.global_variables_initializer()
        ## Pretrain
        # init = tf.global_variables_initializer()
        # sess.run(init)


        # detector_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(detector_loss, var_list=detector_vars)
        print(tl.layers.initialize_global_variables(sess))

        ###============================= LOAD VGG ===============================###
        vgg19_npy_path = "vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")

        npz = np.load(vgg19_npy_path, encoding='latin1').item()

        params = []
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        tl.files.assign_params(sess, params, net_vgg)

        ###============================= TRAINING ===============================###
        ## use first `batch_size` of train set to have a quick test during training
        # sample_imgs = train_hr_imgs[0:batch_size]
        sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path,
                                         n_threads=32)  # if no pre-load train set
        sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
        print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
        sample_imgs_256 = tl.prepro.threading_data(sample_imgs_384.copy(), fn=downsample_fn_3)
        sample_imgs_128 = tl.prepro.threading_data(sample_imgs_384.copy(), fn=downsample_fn_2)
        print('sample LR sub-image:', sample_imgs_256.shape, sample_imgs_256.min(), sample_imgs_256.max())
        tl.vis.save_images(sample_imgs_256, [ni, ni], save_dir_ginit + '/_train_sample_256.png')
        tl.vis.save_images(sample_imgs_128, [ni, ni], save_dir_ginit + '/_train_sample_128.png')
        tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit + '/_train_sample_384.png')
        tl.vis.save_images(sample_imgs_256, [ni, ni], save_dir_gan + '/_train_sample_256.png')
        tl.vis.save_images(sample_imgs_128, [ni, ni], save_dir_gan + '/_train_sample_128.png')
        tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan + '/_train_sample_384.png')

        ###========================= initialize G ====================###
        ## fixed learning rate
        # saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SRGAN_g'))
        sess.run(tf.assign(lr_v, lr_init))
        print(" ** fixed learning rate: %f (for init G)" % lr_init)

        training_ap_calc = APCalculator()
        # td = TrainingData(args.data_dir, 0)

        # initialize_uninitialized_variables(sess)
        for epoch in range(0, n_epoch_init + 1):
            epoch_time = time.time()
            total_mse_loss, n_iter = 0, 0

            # random.shuffle(train_hr_img_list)
            for idx in range(0, len(train_hr_img_list), batch_size):
                step_time = time.time()
                b_imgs_list = train_hr_img_list[idx: idx + batch_size]
                b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
                b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
                b_imgs_128 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn_2)
                b_imgs_256 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn_3)
                ## update G
                errM, _ = sess.run([mse_loss, g_optim_init],
                                   {t_image: b_imgs_128, t_target_image512: b_imgs_384, t_target_image: b_imgs_256})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
                epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                total_mse_loss += errM
                n_iter += 1
                break
            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
            epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
            print(log)

            ## quick evaluation on train set
            if (epoch != 0) and (epoch % 1 == 0):
                out, out512 = sess.run([net_g_test_256.outputs, net_g_test_512.outputs], {t_image: sample_imgs_128})
                print("[*] save images")
                # tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_256_%d.png' % epoch)
                tl.vis.save_images(out512, [ni, ni], save_dir_ginit + '/train_512_%d.png' % epoch)
                # saver1.save(sess, checkpoint_dir + '/SRGAN_X4')

            ## save model
            # if (epoch != 0) and (epoch % 10 == 0):
            #     tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

        ###========================= train GAN (SRGAN) =========================###
        print('[i] Configuring the training data...')
        try:
            print('[i] # training samples:   ', num_train)
            print('[i] # validation samples: ', num_valid)
            print('[i] # classes:            ', num_classes)
            print('[i] Image size:           ', preset.image_size)
        except (AttributeError, RuntimeError) as e:
            print('[!] Unable to load training data:', str(e))
            return 1


        for epoch in range(0, n_epoch + 1):
            print("EPOCH", epoch)
            ## update learning rate
            if epoch != 0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay ** (epoch // decay_every)
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
                print(log)
            elif epoch == 0:
                sess.run(tf.assign(lr_v, lr_init))
                log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
                print(log)


            epoch_time = time.time()
            total_d_loss, total_detector_loss, total_g_loss, n_iter = 0, 0, 0, 0

            # random.shuffle(train_hr_img_list)
            for idx in range(0, len(train_hr_img_list), batch_size):
                step_time = time.time()
                b_imgs_list = train_hr_img_list[idx:idx + batch_size]
                b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
                b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
                b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn_2)
                b_imgs_256 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn_3)

                ##### Detector_Dataset##

                n_train_batches = 1
                global_step = None
                if epoch == 0:
                    lr_values = args.lr_values.split(';')
                    try:
                        lr_values = [float(x) for x in lr_values]
                    except ValueError:
                        print('[!] Learning rate values must be floats')
                        sys.exit(1)

                    lr_boundaries = args.lr_boundaries.split(';')
                    try:
                        lr_boundaries = [int(x) for x in lr_boundaries]
                    except ValueError:
                        print('[!] Learning rate boundaries must be ints')
                        sys.exit(1)

                    ret = compute_lr(lr_values, lr_boundaries)
                    learning_rate, global_step = ret


                # if epoch != 0:
                #
                #     net.build_from_metagraph(metagraph_file, checkpoint_file)
                #     net.build_optimizer_from_metagraph()
                #     print("Hello2")
                # else:
                #     net.build_from_vgg(args.vgg_dir, source.num_classes)
                #     net.build_optimizer(learning_rate=lr_v,
                #                         weight_decay=args.weight_decay,
                #                         momentum=args.momentum)
                #     print("Hello")
                build_from_vgg(args.vgg_dir, source.num_classes,sess,preset)
                build_optimizer(learning_rate=lr_v,
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum)
                print("Hello")

                initialize_uninitialized_variables(sess)
                print("Track")

                # summary_writer = tf.summary.FileWriter(args.tensorboard_dir,
                #                                        sess.graph)

                # saver1 = tf.train.Saver(max_to_keep=10)
                saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ssd'))
                anchors = get_anchors_for_preset(preset)
                training_imgs_samples = []
                validation_imgs_samples = []


                # generator = td.train_generator(idx)
                # y, gt_boxes = td.__batch_generator(train_samples, train_tfs, idx)

                image, y, gt_boxes = gen_batch(train_samples, idx)
                print("Main y shape", y.shape)
                # exit()

                description = '[i] Train {:>2}/{}'.format(epoch + 1, args.epochs)

                # if len(training_imgs_samples) < 3:
                #     saved_images = np.copy(x[:3])
                # for x, y, gt_boxes in tqdm(generator, total=1,
                #                            desc=description, unit='batches'):
                # print(b_imgs_384.shape)
                # print(b_imgs_384.size)
                # feed = {net.image_input: b_imgs_384, net.labels: y,
                #         t_image: b_imgs_96, t_target_image512: b_imgs_384, t_target_image: b_imgs_256}
                #
                # result, loss_batch, _ = sess.run([net.result, net.losses,
                #                                   net.optimizer],
                #                               feed_dict=feed)

                ## update D
                errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_96, t_target_image512: b_imgs_384, t_target_image: b_imgs_256})

                ## update_Detector
                # out512 = sess.run([net_g_test_512.outputs], {t_image: b_imgs_96})



                result, loss_batch, _ = sess.run([net.result, net.losses,
                                                  net.optimizer],
                                                 {net.image_input: b_imgs_384, t_image: b_imgs_96, t_target_image512: b_imgs_384,
                                                  t_target_image: b_imgs_256,
                                                  net.labels: y
                                                  })


                print("Epoch [%2d/%2d] %4d time: %4.4fs, loss: %.8f"
                      %(epoch, n_epoch, n_iter, time.time() - step_time, loss_batch['confidence']))

                if math.isnan(loss_batch['confidence']):
                    print('[!] Confidence loss is NaN.')

                # if epoch == 0: continue

                print("Result Shape", result.shape[0])

                for i in range(result.shape[0]):

                    boxes = decode_boxes(result[i], anchors, 0.5, lid2name)
                    boxes = suppress_overlaps(boxes)
                    training_ap_calc.add_detections(gt_boxes[i], boxes)

                    # if len(training_imgs_samples) < 3:
                    #     training_imgs_samples.append((saved_images[i], boxes))

                    if len(training_imgs_samples) < 3:
                        training_imgs_samples.append((boxes))


                # _,errG, errM, errV, errA, errM_512, errV_512, errA_512, errDetector = sess.run([g_optim, g_loss, mse_loss, vgg_loss, g_gan_loss, mse_loss512, vgg_loss512, g_gan_loss512, net.losses],
                #                                       {net.image_input: b_imgs_384, t_image: b_imgs_96, t_target_image512: b_imgs_384,
                #                                       t_target_image: b_imgs_256,
                #                                       net.labels: y
                #                                       })


                _, errG, errM, errV, errA, errM_512, errV_512, errA_512 = sess.run(
                    [g_optim, g_loss, mse_loss, vgg_loss, g_gan_loss, mse_loss512, vgg_loss512, g_gan_loss512],
                    {net.image_input: b_imgs_384, t_image: b_imgs_96, t_target_image512: b_imgs_384,
                     t_target_image: b_imgs_256, net.labels: y
                    })

                print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f  g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f mse_loss512: %.6f, vgg_loss512: %.6f, g_gan_loss512: %.6f)" %
                      (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA, errM_512, errV_512, errA_512))
                total_d_loss += errD
                print("Here3")
                # total_detector_loss += errDetector
                # total_g_loss += errG
                n_iter += 1
                break
            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
            epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter, total_g_loss / n_iter)
            print(log)

            print("Here4")



            ## quick evaluation on train set
            if (epoch != 0) and (epoch % 1 == 0):
                out, out512 = sess.run([net_g_test_256.outputs, net_g_test_512.outputs], {
                    t_image: sample_imgs_128})
                print("Output 512", out512)

                # ; print('gen sub-image:', out.shape, out.min(), out.max())
                print("[*] save images")
                tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_256_%d.png' % epoch)
                tl.vis.save_images(out512, [ni, ni], save_dir_gan + '/train_512_%d.png' % epoch)

            ## save model
            # if (epoch == 0):
            #     # tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']),
            #     #                   sess=sess)
            #     # tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']),
            #     #                   sess=sess)
            #     # saver1.save(sess, checkpoint_dir + '/SRGAN_X4')
            #     print("Here5")
            #     ckpt_file = "./checkpoint_detector/ssd_train_loss=%.4f.ckpt" % loss_batch['confidence']
            #     saver1.save(sess, ckpt_file)


            if (epoch != 0) and (epoch % 1 == 0):
                print("Here5")
                checkpoint = '{}/e{}.ckpt'.format(args.name, epoch + 1)
                saver1.save(sess, checkpoint)
                print('[i] Checkpoint saved:', checkpoint)

            checkpoint = '{}/final.ckpt'.format(args.name)
            saver1.save(sess, checkpoint)
            print('[i] Checkpoint saved:', checkpoint)

def evaluate():
    print('here 1')
    ## create folders to save result images
    save_dir = "samples/{}_{}".format(tl.global_flag['mode'], in_size)
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g_custom(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (
    size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 256 / in_size, size[1] * 256 / in_size], interp='bicubic',
                                   mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
