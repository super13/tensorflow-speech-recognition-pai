#!/usr/bin/env python

import os
import numpy as np
import time
import warnings
import logging

import tensorflow as tf
from tensorflow.python.ops import ctc_ops

# Custom modules
from features.utils.text import ndarray_to_text, sparse_tuple_to_texts

# in future different than utils class
from models.RNN.utils import create_optimizer
from utilities.set_dirs import get_conf_dir, get_model_dir
from features.utils.text import sparse_tuple_from

# Import the setup scripts for different types of model
from smodels.RNN.rnn import BiRNN as BiRNN_model

logger = logging.getLogger(__name__)


class SpeechTrain(object):
    '''
    Class to train a speech recognition model with TensorFlow in aliyun's pai.

    Requirements:
    - TensorFlow 1.0.1
    - Python 3.5

    Features:
    - Batch loading of input data
    - Checkpoints model
    - Label error rate is the edit (Levenshtein) distance of the top path vs true sentence
    - Logs summary stats for TensorBoard
    - Epoch 1: Train starting with shortest transcriptions, then shuffle

    # Note: All calls to tf.name_scope or tf.summary.* support TensorBoard visualization.

    This class was based on open source code from RNN-Tutorial:
    https://github.com/silicon-valley-data-science/RNN-Tutorial/blob/master/src/train_framework/tf_train_ctc.py

    and it was initially based on open source code from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/DeepSpeech.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    '''

    def __init__(self,
                 model_name=None,
                 debug=False):
        # set TF logging verbosity
        tf.logging.set_verbosity(tf.logging.INFO)

        # Load the configuration file depending on debug True/False
        self.debug = debug
        self.load_configs()

        # set the directories
        self.set_up_directories(model_name)

        # set up the model
        self.set_up_model()

    def load_configs(self):


        self.epochs = 200
        logger.debug('self.epochs = %d', self.epochs)

        self.network_type = 'BiRNN'

        # Number of mfcc features, 13 or 26
        self.n_input = 26

        # Number of contextual samples to include
        self.n_context = 9

        self.model_dir = 'nn/debug_models'

        # set the session name
        self.session_name = '{}_{}'.format(
            self.network_type, time.strftime("%Y%m%d-%H%M%S"))
        sess_prefix_str = 'develop'
        if len(sess_prefix_str) > 0:
            self.session_name = '{}_{}'.format(
                sess_prefix_str, self.session_name)

        # How often to save the model
        self.SAVE_MODEL_EPOCH_NUM = 2

        # decode dev set after N epochs
        self.VALIDATION_EPOCH_NUM = 1

        # decide when to stop training prematurely
        self.CURR_VALIDATION_LER_DIFF = 0.005

        self.AVG_VALIDATION_LER_EPOCHS = 1
        # initialize list to hold average validation at end of each epoch
        self.AVG_VALIDATION_LERS = [
            1.0 for _ in range(self.AVG_VALIDATION_LER_EPOCHS)]

        # setup type of decoder
        self.beam_search_decoder = 'default'

        # determine if the data input order should be shuffled after every epic
        self.shuffle_data_after_epoch = True

        # initialize to store the minimum validation set label error rate
        self.min_dev_ler = 100.0

        # set up GPU if available
        self.tf_device = '/gpu:0'

        # set up the max amount of simultaneous users
        # this restricts GPU usage to the inverse of self.simultaneous_users_count
        self.simultaneous_users_count = 1.3

    def set_up_directories(self, model_name):
        # Set up model directory
        self.model_dir = os.path.join(get_model_dir(), self.model_dir)
        # summary will contain logs
        self.SUMMARY_DIR = os.path.join(
            self.model_dir, "summary", self.session_name)
        # session will contain models
        self.SESSION_DIR = os.path.join(
            self.model_dir, "session", self.session_name)

        if not tf.gfile.Exists(self.SESSION_DIR):
            tf.gfile.MkDir(self.SESSION_DIR)
        if not tf.gfile.Exists(self.SUMMARY_DIR):
            tf.gfile.MkDir(self.SUMMARY_DIR)

        # set the model name and restore if not None
        if model_name is not None:
            self.model_path = os.path.join(self.SESSION_DIR, model_name)
        else:
            self.model_path = None

    def set_up_model(self):
        self.sets = {'train':'train.tfrecords', 'dev':'dev.tfrecords', 'test':'test.tfrecords'}
        self.data_sets={}
        dev={}
        dev['batch_size']=1
        dev['n_examples']=2
        dev['dataset']='dev.tfrecords'
        self.data_sets['dev']=dev

        train={}
        train['batch_size']=1
        train['n_examples']=5
        train['dataset']='train.tfrecords'
        self.data_sets['train']=train

        test={}
        test['batch_size']=1
        test['n_examples']=2
        test['dataset']='test.tfrecords'
        self.data_sets['test']=test


        self.n_examples_train = train['n_examples']
        self.n_examples_dev = dev['n_examples']
        self.n_examples_test = test['n_examples']
        self.batch_size = train['batch_size']
        self.n_batches_per_epoch = int(np.ceil(
            self.n_examples_train / self.batch_size))

        logger.info('''Training model: {}
        Train examples: {:,}
        Dev examples: {:,}
        Test examples: {:,}
        Epochs: {}
        Training batch size: {}
        Batches per epoch: {}'''.format(
            self.session_name,
            self.n_examples_train,
            self.n_examples_dev,
            self.n_examples_test,
            self.epochs,
            self.batch_size,
            self.n_batches_per_epoch))

    def run_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):

            with tf.device(self.tf_device):
                # Run multiple functions on the specificed tf_device
                self.setup_network_and_graph()
                self.load_placeholder_into_network()
                self.setup_loss_function()
                self.setup_optimizer()
                self.setup_decoder()

            self.setup_summary_statistics()

            # create the configuration for the session
            tf_config = tf.ConfigProto()
            tf_config.allow_soft_placement = True
            tf_config.gpu_options.per_process_gpu_memory_fraction = \
                (1.0 / self.simultaneous_users_count)

            # create the session
            self.sess = tf.Session(config=tf_config)

            # initialize the summary writer
            self.writer = tf.summary.FileWriter(
                self.SUMMARY_DIR, graph=self.sess.graph)

            # Add ops to save and restore all the variables
            self.saver = tf.train.Saver()

            # For printing out section headers
            section = '\n{0:=^40}\n'

            # If there is a model_path declared, then restore the model
            if self.model_path is not None:
                self.saver.restore(self.sess, self.model_path)
            # If there is NOT a model_path declared, build the model from scratch
            else:
                # Op to initialize the variables
                init_op = tf.global_variables_initializer()

                # Initializate the weights and biases
                self.sess.run(init_op)

                # MAIN LOGIC for running the training epochs
                logger.info(section.format('Run training epoch'))
                self.run_training_epochs()

            logger.info(section.format('Decoding test data'))
            # make the assumption for working on the test data, that the epoch here is the last epoch
            _, self.test_ler = self.run_batches(self.data_sets['test'], is_training=False,
                                                decode=True, write_to_file=False, epoch=self.epochs)
            # Add the final test data to the summary writer
            # (single point on the graph for end of training run)
            summary_line = self.sess.run(
                self.test_ler_op, {self.ler_placeholder: self.test_ler})
            self.writer.add_summary(summary_line, self.epochs)

            logger.info('Test Label Error Rate: {}'.format(self.test_ler))

            # save train summaries to disk
            self.writer.flush()

            self.sess.close()

    def setup_network_and_graph(self):
        # e.g: log filter bank or MFCC features
        # shape = [batch_size, max_stepsize, n_input + (2 * n_input * n_context)]
        # the batch_size and max_stepsize can vary along each step
        self.input_tensor = tf.placeholder(
            tf.float32, [None, None, self.n_input + (2 * self.n_input * self.n_context)], name='input')

        # Use sparse_placeholder; will generate a SparseTensor, required by ctc_loss op.
        self.targets = tf.sparse_placeholder(tf.int32, name='targets')
        # 1d array of size [batch_size]
        self.seq_length = tf.placeholder(tf.int32, [None], name='seq_length')

    def load_placeholder_into_network(self):
        if self.network_type == 'BiRNN':
            self.logits, summary_op = BiRNN_model(
                self.input_tensor,
                tf.to_int64(self.seq_length),
                self.n_input,
                self.n_context
            )
        else:
            raise ValueError('network_type must be BiRNN')
        self.summary_op = tf.summary.merge([summary_op])

    def setup_loss_function(self):
        with tf.name_scope("loss"):
            self.total_loss = ctc_ops.ctc_loss(
                self.targets, self.logits, self.seq_length,ignore_longer_outputs_than_inputs=True)
            self.avg_loss = tf.reduce_mean(self.total_loss)
            self.loss_summary = tf.summary.scalar("avg_loss", self.avg_loss)

            self.cost_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

            self.train_cost_op = tf.summary.scalar(
                "train_avg_loss", self.cost_placeholder)

    def setup_optimizer(self):
        # Note: The optimizer is created in models/RNN/utils.py
        with tf.name_scope("train"):
            self.optimizer = create_optimizer()
            self.optimizer = self.optimizer.minimize(self.avg_loss)

    def setup_decoder(self):
        with tf.name_scope("decode"):
            if self.beam_search_decoder == 'default':
                self.decoded, self.log_prob = ctc_ops.ctc_beam_search_decoder(
                    self.logits, self.seq_length, merge_repeated=False)
            elif self.beam_search_decoder == 'greedy':
                self.decoded, self.log_prob = ctc_ops.ctc_greedy_decoder(
                    self.logits, self.seq_length, merge_repeated=False)
            else:
                logging.warning("Invalid beam search decoder option selected!")

    def setup_summary_statistics(self):
        # Create a placholder for the summary statistics
        with tf.name_scope("accuracy"):
            # Compute the edit (Levenshtein) distance of the top path
            distance = tf.edit_distance(
                tf.cast(self.decoded[0], tf.int32), self.targets)

            # Compute the label error rate (accuracy)
            self.ler = tf.reduce_mean(distance, name='label_error_rate')
            self.ler_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
            self.train_ler_op = tf.summary.scalar(
                "train_label_error_rate", self.ler_placeholder)
            self.dev_ler_op = tf.summary.scalar(
                "validation_label_error_rate", self.ler_placeholder)
            self.test_ler_op = tf.summary.scalar(
                "test_label_error_rate", self.ler_placeholder)

    def run_training_epochs(self):
        train_start = time.time()
        for epoch in range(self.epochs):
            # Initialize variables that can be updated
            save_dev_model = False
            stop_training = False
            is_checkpoint_step, is_validation_step = \
                self.validation_and_checkpoint_check(epoch)

            epoch_start = time.time()

            self.train_cost, self.train_ler = self.run_batches(
                self.data_sets['train'],
                is_training=True,
                decode=False,
                write_to_file=False,
                epoch=epoch)

            epoch_duration = time.time() - epoch_start

            log = 'Epoch {}/{}, train_cost: {:.3f}, \
                   train_ler: {:.3f}, time: {:.2f} sec'
            logger.info(log.format(
                epoch + 1,
                self.epochs,
                self.train_cost,
                self.train_ler,
                epoch_duration))

            summary_line = self.sess.run(
                self.train_ler_op, {self.ler_placeholder: self.train_ler})
            self.writer.add_summary(summary_line, epoch)

            summary_line = self.sess.run(
                self.train_cost_op, {self.cost_placeholder: self.train_cost})
            self.writer.add_summary(summary_line, epoch)



            # Run validation if it was determined to run a validation step
            if is_validation_step:
                self.run_validation_step(epoch)

            # if (epoch + 1) == self.epochs or is_checkpoint_step:
            #     # save the final model
            #     save_path = self.saver.save(self.sess, os.path.join(
            #         self.SESSION_DIR, 'model.ckpt'), epoch)
            #     logger.info("Model saved: {}".format(save_path))

            if save_dev_model:
                # If the dev set is not improving,
                # the training is killed to prevent overfitting
                # And then save the best validation performance model
                save_path = self.saver.save(self.sess, os.path.join(
                    self.SESSION_DIR, 'model-best.ckpt'))
                logger.info(
                    "Model with best validation label error rate saved: {}".
                    format(save_path))


            if stop_training:
                break

        train_duration = time.time() - train_start
        logger.info('Training complete, total duration: {:.2f} min'.format(
            train_duration / 60))

    def read_and_decode(self,filename): # 读入train.tfrecords
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
        source_lengths = features['source_lengths']
        label = features['label']
        return source,source_lengths, label

    def run_validation_step(self, epoch):
        dev_ler = 0

        _, dev_ler = self.run_batches(self.data_sets['dev'],
                                      is_training=False,
                                      decode=True,
                                      write_to_file=False,
                                      epoch=epoch)

        logger.info('Validation Label Error Rate: {}'.format(dev_ler))

        summary_line = self.sess.run(
            self.dev_ler_op, {self.ler_placeholder: dev_ler})
        self.writer.add_summary(summary_line, epoch)

        if dev_ler < self.min_dev_ler:
            self.min_dev_ler = dev_ler

        # average historical LER
        history_avg_ler = np.mean(self.AVG_VALIDATION_LERS)

        # if this LER is not better than average of previous epochs, exit
        if history_avg_ler - dev_ler <= self.CURR_VALIDATION_LER_DIFF:
            log = "Validation label error rate not improved by more than {:.2%} \
                  after {} epochs. Exit"
            warnings.warn(log.format(self.CURR_VALIDATION_LER_DIFF,
                                     self.AVG_VALIDATION_LER_EPOCHS))

        # save avg validation accuracy in the next slot
        self.AVG_VALIDATION_LERS[
            epoch % self.AVG_VALIDATION_LER_EPOCHS] = dev_ler

    def validation_and_checkpoint_check(self, epoch):
        # initially set at False unless indicated to change
        is_checkpoint_step = False
        is_validation_step = False

        # Check if the current epoch is a validation or checkpoint step
        if (epoch > 0) and ((epoch + 1) != self.epochs):
            if (epoch + 1) % self.SAVE_MODEL_EPOCH_NUM == 0:
                is_checkpoint_step = True
            if (epoch + 1) % self.VALIDATION_EPOCH_NUM == 0:
                is_validation_step = True

        return is_checkpoint_step, is_validation_step

    def run_batches(self, dataset, is_training, decode, write_to_file, epoch):
        n_examples = dataset['n_examples']

        n_batches_per_epoch = int(np.ceil(n_examples / dataset['batch_size']))

        self.train_cost = 0
        self.train_ler = 0

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3*4

        source,source_lengths,label=self.read_and_decode(dataset['dataset'])
        data_batch = tf.train.shuffle_batch([source,source_lengths,label], batch_size=dataset['batch_size'],
                                           num_threads=8,
                                           capacity=capacity,
                                       min_after_dequeue=min_after_dequeue)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=self.sess)

        for batch in range(n_batches_per_epoch):
            # Get next batch of training data (audio features) and transcripts
            dats = self.sess.run([data_batch])
            dat=dats[0]
            #source, source_lengths, sparse_labels = dataset.next_batch()
            #print(np.shape(source))

            feature_tensor = tf.sparse_tensor_to_dense(dat[0], default_value=0)
            source = tf.reshape(feature_tensor, [dataset['batch_size'],-1, 494]).eval(session=self.sess)
            source_lengths=dat[1]
            feature_labels =tf.sparse_tensor_to_dense(dat[2], default_value=0)
            labels =tf.reshape(feature_labels, [dataset['batch_size'],-1]).eval(session=self.sess)
            sparse_labels = sparse_tuple_from(labels)
            # print(type(sparse_labels),"dddddddddddddd")
            # print(sparse_labels,"dddddddddddddd")

            feed = {self.input_tensor: source,
                    self.targets: sparse_labels,
                    self.seq_length: source_lengths}

            # If the is_training is false, this means straight decoding without computing loss
            if is_training:
                # avg_loss is the loss_op, optimizer is the train_op;
                # running these pushes tensors (data) through graph
                batch_cost, _ = self.sess.run(
                    [self.avg_loss, self.optimizer], feed)
                self.train_cost += batch_cost * dataset['batch_size']
                logger.debug('Batch cost: %.2f | Train cost: %.2f | Batch :%d | Total batch per epoch:%d | Epoch: %d',
                             batch_cost, self.train_cost,batch,n_batches_per_epoch,epoch)

            self.train_ler += self.sess.run(self.ler, feed_dict=feed) * dataset['batch_size']
            logger.debug('Label error rate: %.2f', self.train_ler)

            # Turn on decode only 1 batch per epoch
            if decode and batch == 0:
                d = self.sess.run(self.decoded[0], feed_dict={
                    self.input_tensor: source,
                    self.targets: sparse_labels,
                    self.seq_length: source_lengths}
                )
                dense_decoded = tf.sparse_tensor_to_dense(
                    d, default_value=-1).eval(session=self.sess)
                dense_labels = sparse_tuple_to_texts(sparse_labels)

                # only print a set number of example translations
                counter = 0
                counter_max = 4
                if counter < counter_max:
                    for orig, decoded_arr in zip(dense_labels, dense_decoded):
                        # convert to strings
                        decoded_str = ndarray_to_text(decoded_arr)
                        logger.info('Batch {}, file {}'.format(batch, counter))
                        logger.info('Original: {}'.format(orig))
                        logger.info('Decoded:  {}'.format(decoded_str))
                        counter += 1

                # save out variables for testing
                self.dense_decoded = dense_decoded
                self.dense_labels = dense_labels

        # Metrics mean
        if is_training:
            self.train_cost /= n_examples
        self.train_ler /= n_examples

        # Populate summary for histograms and distributions in tensorboard
        self.accuracy, summary_line = self.sess.run(
            [self.avg_loss, self.summary_op], feed)
        self.writer.add_summary(summary_line, epoch)

        return self.train_cost, self.train_ler


# to run in console
if __name__ == '__main__':
    import click

    # Use click to parse command line arguments
    @click.command()
    @click.option('--config', default='neural_network.ini', help='Configuration file name')
    @click.option('--name', default=None, help='Model name for logging')
    @click.option('--debug', type=bool, default=False,
                  help='Use debug settings in config file')
    # Train RNN model using a given configuration file
    def main(config='neural_network.ini', name=None, debug=False):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        global logger
        logger = logging.getLogger(os.path.basename(__file__))

        # create the Tf_train_ctc class
        tf_train_ctc = SpeechTrain( model_name=name, debug=debug)

        # run the training
        tf_train_ctc.run_model()

    main()