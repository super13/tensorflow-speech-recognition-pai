#!/usr/bin/env python
import os
import sys
import tensorflow as tf
import numpy as np
from features.utils.load_audio_to_mem import get_audio_and_transcript, pad_sequences
from features.utils.text import sparse_tuple_from

txt_files = []
for root, dirs, files in os.walk(sys.argv[1]):
    for fname in files:
        if fname.endswith('.txt'):
            txt_files.append(os.path.join(root,fname))


wav_files = [x.replace('.txt', '.wav') for x in txt_files]
(source, _, target, _) = get_audio_and_transcript(txt_files,wav_files,26,9)
source, source_lengths = pad_sequences(source)

writer = tf.python_io.TFRecordWriter(sys.argv[2])
for sa,la,ta in zip(source,source_lengths,target):
    a_reshape = np.reshape(sa, -1)
    print(ta)
    print(np.shape(ta))
    print(type(ta))
    example = tf.train.Example(features=tf.train.Features(feature={
    'source': tf.train.Feature(float_list=tf.train.FloatList(value=a_reshape)),
    'source_lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=[la])),
    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=ta))
    }))
    writer.write(example.SerializeToString())  #序列化为字符串
