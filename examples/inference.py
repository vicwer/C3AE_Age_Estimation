#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from models.run_net import C3AENet
from config import cfg
import cv2
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def inference(img_path, epoch):
    is_training = False
    cfg.batch_size = 1
    ckpt_dir = cfg.ckpt_path

    configer = tf.ConfigProto()
    configer.gpu_options.per_process_gpu_memory_fraction = 0.1
    with tf.Session(config=configer) as sess:
        imgs_holder = tf.placeholder(tf.float32, shape=[3, 64, 64, 3])
        model = C3AENet(imgs_holder, None, None, is_training)
        age = model.predict()

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, ckpt_dir + 'C3AEDet-' + str(epoch))
        sess.run(tf.local_variables_initializer())

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        center_x, center_y = w // 2, h // 2
        crop_img_1 = img[center_y-100:center_y+100, center_x-100:center_x+100, :]
        crop_img_2 = img[center_y-80:center_y+80, center_x-80:center_x+80, :]
        crop_img_3 = img[center_y-60:center_y+60, center_x-60:center_x+60, :]
        crop_img_1 = cv2.resize(crop_img_1, (64,64))
        crop_img_2 = cv2.resize(crop_img_2, (64,64))
        crop_img_3 = cv2.resize(crop_img_3, (64,64))
        merge_img = np.asarray([crop_img_1, crop_img_2, crop_img_3], dtype=np.float32) / 255.

        age_res = sess.run(age, feed_dict={imgs_holder: np.reshape(merge_img, [3, 64, 64, 3])})
        print('-----age: {}-----'.format(age_res))

    tf.reset_default_graph()

if __name__ == '__main__':
    img_path = sys.argv[1]
    epoch = 12
    print('-----epoch: {}-----'.format(epoch))
    inference(img_path, epoch)
