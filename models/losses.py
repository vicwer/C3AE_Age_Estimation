#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from config import cfg

def l1_loss(preds, labels):
    return tf.reduce_sum(tf.abs(labels - preds))

def kl_loss(preds, labels, l1):
    return -(tf.reduce_sum(tf.reduce_sum(labels * tf.log(preds + 1e-10)), axis=1)) + l1

