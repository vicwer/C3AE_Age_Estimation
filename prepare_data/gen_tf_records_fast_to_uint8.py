# encoding: utf-8

import numpy as np
import tensorflow as tf
import os
import cv2
from tqdm import tqdm
import re
import sys
sys.path.append('..')
from config import cfg

def load_file(file_path):
    '''
    load imgs_path, classes and labels
    '''
    imgs_path = []
    age_labels = []
    age_vectors = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_path = line.strip().split(' ')[0]
            age_label = float(line.strip().split(' ')[1])
            age_vector = float(line.strip().split(' ')[2:])
            imgs_path.append(img_path)
            age_labels.append(age_label)
            age_vectors.append(age_vector)

    return np.asarray(imgs_path), np.asarray(age_labels), np.asarray(age_vectors)

def extract_image(image_path, height, width, is_resize=True):
    '''
    get b->g->r image data
    '''
    img = cv2.imread(image_path)
    if is_resize:
        h, w, _ = img.shape
        image = cv2.resize(img, (96, 96))
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
    else:
        image = img
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
    image_data = np.array(image, dtype='uint8')
    return image_data

def run_encode(file_path, tf_records_filename):
    '''
    encode func
    '''
    imgs_path, age_labels, age_vectors = load_file(file_path)
    height, width = 220, 220
    imgs = []
    writer = tf.python_io.TFRecordWriter(tf_records_filename)
    for i in tqdm(range(imgs_path.shape[0])):
        img = extract_image(imgs_path[i], height, width, is_resize=False)
        img = img.tostring()
        age_label = age_labels[i].flatten().tolist()
		age_vector = age_vectors[i].flatten().tolist()
        example = tf.train.Example(features=tf.train.Features(feature={
                      'age_label' : tf.train.Feature(float_list = tf.train.FloatList(value=age_label)),
                      'age_vector' : tf.train.Feature(float_list = tf.train.FloatList(value=age_vector))
                      'feature': tf.train.Feature(bytes_list = tf.train.BytesList(value=[img]))
                  }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    file_path = '../data/train_list/train.txt'
    tf_records_filename = '../tf_records/train.records'

    run_encode(file_path, tf_records_filename)
