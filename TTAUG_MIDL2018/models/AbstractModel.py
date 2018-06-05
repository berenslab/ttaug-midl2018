from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf


class AbstractModel(object):
    """
    Abstract (base) model to represent a basic model structure.
    """

    __metaclass__ = ABCMeta
    type = 'AbstractModel'

    def __init__(self, instance_shape, num_classes, name='Model'):

        self.instance_shape = instance_shape
        self.num_inputs = np.prod(self.instance_shape)
        self.num_classes = num_classes
        self.name = name
        self.diagnostics = {}
        self.session = None
        # Some nodes of interest
        self.logits = None
        self.penultimate_features = None
        self.predictions = None
        self.predictions_1hot = None
        self.loss = None
        self.learning_rate = None
        self.train_op = None
        self.saver = None

        self.descriptor = None
        self.model_path = None

        # print('Instance shape : ' + str(self.instance_shape))

        batch_shape = [None]
        for num in instance_shape:
            batch_shape.append(num)
        # print('Batch shape : ' + str(batch_shape))

        with tf.name_scope('layer0'):
            self.inputs = tf.placeholder(dtype=tf.float32, shape=batch_shape, name='inputs')
            self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='labels')
            self.labels_1hot = tf.squeeze(tf.one_hot(indices=self.labels, depth=num_classes, name='labels_1hot'))

    def feed_dict(self, reader, batch_size, normalize):
        x_batch, y_batch, _ = reader.next_batch(batchSize=batch_size, normalize=normalize)
        feed_dict = {self.inputs: x_batch,
                     self.labels: y_batch}
        return feed_dict

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    @abstractmethod
    def finalize(self):
        pass
