#!/usr/bin/env python
# -*- coding: utf-8 -*- #

import os
import sys
import tensorflow as tf
import numpy as np
import pickle
import random
from features.utils.text import sparse_tuple_from
from features.utils.load_audio_to_mem import get_audio_and_transcript, \
    pad_sequences


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


txt_files = []
for root, dirs, files in os.walk(sys.argv[1]):
    for fname in files:
        if fname.endswith('.txt'):
            txt_files.append(os.path.join(root, fname))


if sys.argv[2].find('train') >= 0:
    txt_files = sorted(txt_files, key=os.path.getsize, reverse=False)
else:
    random.shuffle(txt_files)


chunks_list = chunks(txt_files, 16)
index = 0
for txt_file in chunks_list:
    wav_files = [x.replace('.txt', '.wav') for x in txt_file]
    (source, _, target, _) = get_audio_and_transcript(
        txt_file, wav_files, 26, 9)
    source, source_lengths = pad_sequences(source)
    print("fff", np.shape(source))
    print("ffd", np.shape(source_lengths))
    d = {}
    d['source'] = source
    d['source_lengths'] = source_lengths
    d['label'] = sparse_tuple_from(target)
    tf.gfile.MkDir(sys.argv[2])
    print("index:", index)
    output = open(sys.argv[2]+'/pk_data' + str(index), 'wb')
    index += 1
    pickle.dump(d, output, protocol=2)
