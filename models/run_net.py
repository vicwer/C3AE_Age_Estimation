#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from models.network import Network
from config import cfg
from models.losses import l1_loss, kl_loss

class C3AENet:
    def __init__(self, img, age_labels, age_vector, is_training, batcn_norm_decay=0.997):
        self.img = img
        self.age_labels = age_labels
        self.age_vector = age_vector
        self.is_training = is_training
        self.batch_norm_decay = batcn_norm_decay
        self.img_shape = tf.shape(self.img)
        backbone = Network()
        if is_training:
            self.feats, self.pred, self.l1_loss = backbone.inference(self.is_training, self.img)
        else:
            self.head, self.conv = backbone.inference(self.is_training, self.img)

    def compute_loss(self):
        with tf.name_scope('loss_0'):
            loss_l1 = l1_loss(self.pred, self.age_labels)
			loss_kl = kl_loss(self.feats, self.age_vector, self.l1_loss)
            self.all_loss = loss_l1 + loss_kl
        return self.all_loss

    def predict(self):
        '''
        only support single image prediction, TODO...
        '''
        pred_score = tf.reshape(self.head, (-1, cfg.classes))
        score = tf.nn.softmax(tf.reshape(self.head, (-1, cfg.classes)))
        class_index = tf.argmax(pred_score, 1)
        return class_index, score, self.conv
