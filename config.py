from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
import os

cfg = edict()

cfg.batch_size = 60
# cfg.batch_size = 256

cfg.data_path = '../tf_records/train.records'
cfg.ckpt_path = '../ckpt/'

# training options
cfg.train = edict()

cfg.train.ignore_thresh = .5
cfg.train.ratio = 0.8
cfg.train.momentum = 0.9
cfg.train.bn_training = True
cfg.train.weight_decay = 0.00001 # 0.00004
cfg.train.learning_rate = [1e-3, 1e-4, 1e-5]
cfg.train.max_batches = 50000 # 63000
cfg.train.lr_steps = [10000., 20000.]
cfg.train.lr_scales = [.1, .1]
cfg.train.num_gpus = 4
cfg.train.tower = 'tower'

cfg.train.learn_rate = 0.001
cfg.train.learn_rate_decay = 0.9
cfg.train.learn_rate_decay_epoch = 2
cfg.train.num_samples = 150000
cfg.epochs = 160
cfg.PRINT_LAYER_LOG = True
