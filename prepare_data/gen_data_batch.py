#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')
from config import cfg
import os
import re
import cv2

def parser(example):
    feats = tf.parse_single_example(example, features={'age_label': tf.FixedLenFeature([1], tf.float32),
                                                       'age_vector': tf.FixedLenFeature([12], tf.float32)
                                                       'feature': tf.FixedLenFeature([], tf.string)})
    age_label = feats['age_label']
	age_vector = feats['age_vector']

    img = tf.decode_raw(feats['feature'], tf.uint8)
    img = tf.reshape(img, [220, 220, 3])
    img = tf.image.random_flip_left_right(img)

    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_brightness(img, max_delta=0.05)

    img_crop_1 = tf.image.resize_image_with_crop_or_pad(img, 200, 200)
    img_crop_1 = tf.image.resize_images(img_crop_1, [64, 64])
    img_crop_1 = tf.cast(img_crop_1, tf.float32) / 255.0

    img_crop_2 = tf.image.resize_image_with_crop_or_pad(img, 160, 160)
    img_crop_2 = tf.image.resize_images(img_crop_2, [64, 64])
    img_crop_2 = tf.cast(img_crop_2, tf.float32) / 255.0

    img_crop_3 = tf.image.resize_image_with_crop_or_pad(img, 120, 120)
    img_crop_3 = tf.image.resize_images(img_crop_3, [64, 64])
    img_crop_3 = tf.cast(img_crop_3, tf.float32) / 255.0

    img = tf.stack([img_crop_1, img_crop_2, img_crop_3], axis=0)
    print(img.get_shape())

    return img, age_label, age_vector

def gen_data_batch(tf_records_filename, batch_size):
    dt = tf.data.TFRecordDataset(tf_records_filename)
    #dt = dt.map(parser, num_parallel_calls=4)
    dt = dt.map(parser, num_parallel_calls=4)
    dt = dt.prefetch(batch_size)
    dt = dt.shuffle(buffer_size=24*batch_size)
    dt = dt.repeat()
    dt = dt.batch(batch_size)
    iterator = dt.make_one_shot_iterator()
    imgs, age_label, age_vector = iterator.get_next()

    return imgs, age_label, age_vector

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    tf_records_filename = cfg.data_path

    imgs, age_label, age_vector = gen_data_batch(tf_records_filename, cfg.batch_size)
    print(imgs.get_shape())
    imgs = tf.reshape(imgs, (-1, imgs.get_shape()[2], imgs.get_shape()[3], imgs.get_shape()[4]))
    imgs_split = tf.split(imgs, cfg.train.num_gpus)
    age_label_split = tf.split(age_label, cfg.train.num_gpus)
	age_vector_split = tf.split(age_vector, cfg.num_gpus)

    configer = tf.ConfigProto()
    configer.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess=tf.Session(config=configer)
    for i in range(20):
        for j in range(cfg.train.num_gpus):
            imgs_, age_label_ = sess.run([imgs_split[j], age_label_split[j]])
            # print(imgs_[0, :, :, :])
            print(imgs_.shape)

            for k in range(imgs_.shape[0]):
                cv2.imshow('img', imgs_[k].astype(np.uint8))
                cv2.waitKey(0)
                print(age_label_.shape)
