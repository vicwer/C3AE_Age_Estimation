#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from config import cfg

def l1_loss(preds, labels):
    preds = tf.reshape(preds, (cfg.batch_size,))
    # print("preds", preds.get_shape())
    labels = tf.reshape(labels, (cfg.batch_size,))
    # print("labels", labels.get_shape())
    l1_loss = tf.abs(labels - preds)
    # print("l1_loss", l1_loss.get_shape())
    _, k_idx = tf.nn.top_k(l1_loss, tf.cast(cfg.batch_size * cfg.ohem_ratio, tf.int32))
    loss = tf.gather(l1_loss, k_idx)
    return tf.reduce_mean(loss)

def kl_loss(preds, labels, l1):
    # h_pq = -(tf.reduce_sum(tf.reduce_sum(labels * tf.log(preds + 1e-10), axis=1)))
    # h_p = -tf.reduce_sum(labels * tf.log(labels + 1e-10))
    # loss = h_pq - h_p + l1
    kl_loss = tf.reduce_sum(labels * tf.log(preds + 1e-10), axis=1)
    # print("kl_loss", kl_loss.get_shape())
    kl_loss = tf.reshape(kl_loss, (cfg.batch_size,))
    # print("kl_loss", kl_loss.get_shape())
    _, k_idx = tf.nn.top_k(kl_loss, tf.cast(cfg.batch_size * cfg.ohem_ratio, tf.int32))
    loss = tf.gather(kl_loss, k_idx)
    return -tf.reduce_mean(loss) + l1
