#!/usr/bin/env python
import os
import sys
import tensorflow as tf
import numpy as np
import time


def read_and_decode(filename): # 读入train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'source' : tf.VarLenFeature(dtype=tf.float32),
                                           'source_lengths' : tf.FixedLenFeature([], tf.int64),
                                           'label': tf.VarLenFeature(dtype=tf.int64),
                                       })

    source = features['source']
    print(source,"ffffffff")
    source_lengths = features['source_lengths']
    print(source_lengths,"ddddddddddddd")
    label = features['label']
    return source,source_lengths, label

source,source_lengths,label=read_and_decode(sys.argv[1])

with tf.Session() as sess:
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3*4

    source,source_lengths,label=read_and_decode(sys.argv[1])
    img_batch = tf.train.shuffle_batch([source,source_lengths,label], batch_size=16,
                                       num_threads=8,
                                       capacity=capacity,
                                   min_after_dequeue=min_after_dequeue)


    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    i = 0
    read_tfrecord_start_time = time.time()
    try:
        #while not coord.should_stop():
        imgs = sess.run([img_batch])
        for img in imgs:
            print(np.shape(img[0]))
            feature_tensor = tf.sparse_tensor_to_dense(img[0], default_value=0)
            print(np.shape(feature_tensor))
            feature_tensor = tf.reshape(feature_tensor, [16,-1, 494])
            print(feature_tensor)
            print(img[2])

    finally:
        coord.request_stop()
    coord.join(threads)
    read_tfrecord_duration = time.time() - read_tfrecord_start_time
    print("Read TFrecord Duration:   %.3f" % read_tfrecord_duration)
