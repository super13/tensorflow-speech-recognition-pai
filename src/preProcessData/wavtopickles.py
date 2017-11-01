#!/usr/bin/env python
# -*- coding: utf-8 -*- #

import os
import sys
import tensorflow as tf
import numpy as np
import pickle
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

chunks_list = chunks(txt_files, 16)
index = 0
for txt_file in chunks_list:
    wav_files = [x.replace('.txt', '.wav') for x in txt_file]
    (source, _, target, _) = get_audio_and_transcript(
        txt_file, wav_files, 26, 9)
    source, source_lengths = pad_sequences(source)

    da = []
    dl = []
    for sa, la in zip(source, source_lengths):
        a_reshape = np.reshape(sa, -1)
        # print(ta)
        da.append(sa)
        dl.append(la)
    d = {}
    d['source'] = da
    d['source_lengths'] = dl
    d['label'] = sparse_labels = sparse_tuple_from(target)
    tf.gfile.MkDir(sys.argv[2])
    print("index:", index)
    output = open(sys.argv[2]+'/pk_data' + str(index), 'wb')
    index += 1
    pickle.dump(d, output)
