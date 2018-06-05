import numpy as np
import tensorflow as tf

from models.AbstractModel import AbstractModel
from utils.TensorUtil import resnet_bottleneck_head_block
from utils.TensorUtil import resnet_bottleneck_identity_block
from utils.TensorUtil import resnet_bottleneck_head_block_preactivation
from utils.TensorUtil import resnet_bottleneck_identity_block_preactivation
from utils.TensorUtil import conv_prelu
from utils.TensorUtil import conv_prelu_preactivation
from utils.TensorUtil import max_pool
from utils.TensorUtil import avg_pool
from utils.TensorUtil import fc_prelu
from utils.TensorUtil import fc_prelu_preactivation
from utils.DataAugmentation import do_nothing, data_augmentation
from utils.Reader import AdvancedReader

from sklearn.metrics import roc_curve, auc


class ResNet50(AbstractModel):

    type = 'ResNet50'

    def __init__(self, instance_shape, num_classes, name='ResNet50'):
        super().__init__(instance_shape=instance_shape, num_classes=num_classes, name=name)
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        # self.pos_weights = tf.placeholder(dtype=tf.float32, shape=[1, self.num_classes])

        # Below parameters are for BatchRenorm: they will be used for an affine transformation of normalized activations
        self.rmin = tf.placeholder(dtype=tf.float32, shape=[1], name='renorm_clip_rmin')
        self.rmax = tf.placeholder(dtype=tf.float32, shape=[1], name='renorm_clip_rmax')
        self.dmax = tf.placeholder(dtype=tf.float32, shape=[1], name='renorm_clip_dmax')

        self.momentum = tf.placeholder(dtype=tf.float32, shape=[], name='momentum_coefficient')

    def feed_dict(self, reader, batch_size, normalize, is_train, sampling, iter, max_iter):
        x_batch, y_batch, _ = reader.next_batch(batch_size=batch_size, normalize=normalize,
                                                shuffle=is_train, sampling=sampling
                                                )

        progress = float(iter) / float(max_iter)
        if progress < 0.01:  # up to this point, use BatchNorm alone
            rmax = 1.
            rmin = 1.
            dmax = 0.
        else:  # then, gradually increase the clipping values
            rmax = np.exp(2. * progress)  # 1.5
            rmin = 1. / rmax
            dmax = np.exp(2.5 * progress) - 1  # 2.
        if progress > 0.95:
            rmin = 0.

        # momentum settings
        # momentum_max = 0.95
        # momentum = 1 - np.power(2, -1-np.log2(np.floor(iter/250.)+1))

        feed_dict = {self.inputs: x_batch,
                     self.labels: y_batch,
                     # self.pos_weights: pos_weights,
                     self.is_training: is_train,
                     self.rmin: [rmin],
                     self.rmax: [rmax],
                     self.dmax: [dmax],
                     # self.momentum: np.minimum(1. - (3. / (iter+5.)), 0.95)
                     # self.momentum: np.minimum(momentum, momentum_max)
                     }
        return feed_dict

    def build(self, conv_stack_depths=[3, 4, 6, 3],
              num_filters=[[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]],
              fc_depths=[1024],
              _lambda=0.001, learning_rate=0.001, decay_steps=100, decay_rate=0.96,
              data_aug=True, use_batch_renorm=False, preactivation=True):

        assert len(conv_stack_depths) == len(num_filters), \
            'Convolutional stack depths do not match the number of filters per stack'

        self.descriptor = 'regConst_' + str(_lambda) + 'lr_' + str(learning_rate)
        if use_batch_renorm:
            self.descriptor = self.descriptor + 'BatchReNorm_'
        else:
            self.descriptor = self.descriptor + 'BatchNorm_'

        if preactivation:
            self.descriptor = self.descriptor + 'preactivation'
        else:
            self.descriptor = self.descriptor + 'original'
        self.model_path = './modelstore/' + self.name + self.descriptor + '.ckpt'

        # branching for data augmentation operations in training
        if data_aug:
            inputs2next = tf.cond(self.is_training, true_fn=lambda: data_augmentation(self.inputs),
                                  false_fn=lambda: do_nothing(self.inputs), name='data_augmentation_if_training'
                                  )
        else:
            inputs2next = self.inputs

        scope = 'conv1'  # the very first 5x5 convolution and pooling operations on inputs
        with tf.variable_scope(scope):
            print(scope)
            if preactivation:
                inputs2next = conv_prelu_preactivation(inputs=inputs2next,
                                                       kernel_shape=[5, 5], num_filters=num_filters[0][0], strides=[2, 2],
                                                       reg_const=_lambda,
                                                       is_training=self.is_training,
                                                       use_batch_renorm=use_batch_renorm, rmin=self.rmin, rmax=self.rmax,
                                                       dmax=self.dmax
                                                       )
            else:
                inputs2next = conv_prelu(inputs=inputs2next,
                                         kernel_shape=[5, 5], num_filters=num_filters[0][0], strides=[2, 2],
                                         reg_const=_lambda,
                                         is_training=self.is_training,
                                         use_batch_renorm=use_batch_renorm, rmin=self.rmin, rmax=self.rmax, dmax=self.dmax
                                         )
            inputs2next = max_pool(inputs2next, kernel_shape=[3, 3], strides=[2, 2], padding='SAME')

        # Iterate over convolutional stacks
        for stack_idx in range(len(conv_stack_depths)):
            scope = 'conv' + str(stack_idx+2)
            with tf.variable_scope(scope):  # e.g., conv2
                head_placed = False
                # Iterate over residual block in each conv. stack
                for i in range(conv_stack_depths[stack_idx]):
                    with tf.variable_scope(str(i + 1)):  # e.g., conv2/1
                        print(scope + '/' + str(i + 1))
                        if not head_placed:  # Place a head block first.
                            if stack_idx == 0:
                                head_conv_stride = [1, 1]
                            else:
                                head_conv_stride = [2, 2]
                            if preactivation:
                                inputs2next = resnet_bottleneck_head_block_preactivation(inputs2next,
                                                                                         kernel_shapes=[[1, 1], [3, 3], [1, 1]],
                                                                                         num_filters=num_filters[stack_idx],
                                                                                         strides=head_conv_stride, padding='SAME',
                                                                                         reg_const=_lambda,
                                                                                         is_training=self.is_training,
                                                                                         use_batch_renorm=use_batch_renorm,
                                                                                         rmin=self.rmin, rmax=self.rmax,
                                                                                         dmax=self.dmax
                                                                                         )
                            else:
                                inputs2next = resnet_bottleneck_head_block(inputs2next,
                                                                           kernel_shapes=[[1, 1], [3, 3], [1, 1]],
                                                                           num_filters=num_filters[stack_idx],
                                                                           strides=head_conv_stride, padding='SAME',
                                                                           reg_const=_lambda,
                                                                           is_training=self.is_training, use_batch_renorm=use_batch_renorm,
                                                                           rmin=self.rmin, rmax=self.rmax, dmax=self.dmax
                                                                           )
                            head_placed = True
                        else:
                            if preactivation:
                                inputs2next = resnet_bottleneck_identity_block_preactivation(inputs2next,
                                                                                             kernel_shapes=[[1, 1], [3, 3], [1, 1]],
                                                                                             num_filters=num_filters[stack_idx],
                                                                                             strides=[1, 1], padding='SAME',
                                                                                             reg_const=_lambda,
                                                                                             is_training=self.is_training,
                                                                                             use_batch_renorm=use_batch_renorm,
                                                                                             rmin=self.rmin, rmax=self.rmax,
                                                                                             dmax=self.dmax
                                                                                             )
                            else:
                                inputs2next = resnet_bottleneck_identity_block(inputs2next,
                                                                               kernel_shapes=[[1, 1], [3, 3], [1, 1]],
                                                                               num_filters=num_filters[stack_idx],
                                                                               strides=[1, 1], padding='SAME', reg_const=_lambda,
                                                                               is_training=self.is_training, use_batch_renorm=use_batch_renorm,
                                                                               rmin=self.rmin, rmax=self.rmax, dmax=self.dmax
                                                                               )

        # concatenate features from global max pooling and global avg. pooling
        # and flatten them before the FC layers, or logits if no FC layers is present.
        inputs2next = tf.concat([tf.layers.flatten(max_pool(inputs2next, kernel_shape=[16, 16],
                                                            strides=[1, 1], padding='VALID'),
                                                   name='flattened_max_pool_feat'),
                                 tf.layers.flatten(avg_pool(inputs2next, kernel_shape=[16, 16],
                                                            strides=[1, 1], padding='VALID'),
                                                   name='flattened_avg_pool_feat')
                                 ],
                                axis=-1,
                                name='concat_glob_max_glob_avg')

        fan_in = inputs2next.get_shape().as_list()[-1]  # num_filters[-1][-1]
        for fc_idx in range(len(fc_depths)):
            scope = 'fc' + str(fc_idx+1)
            print(scope)
            with tf.variable_scope(scope):
                fan_out = fc_depths[fc_idx]
                if preactivation:
                    # inputs2next = tf.matmul(inputs2next, tf.eye(fan_in)) # With TF 1.4, this hack was useful.
                    inputs2next = fc_prelu_preactivation(inputs=inputs2next, fan_in=fan_in, fan_out=fan_out,
                                                         reg_const=_lambda, is_training=self.is_training,
                                                         use_batch_renorm=use_batch_renorm,
                                                         rmin=self.rmin, rmax=self.rmax, dmax=self.dmax
                                                         )
                else:
                    inputs2next = fc_prelu(inputs=inputs2next, fan_in=fan_in, fan_out=fan_out,
                                           reg_const=_lambda, is_training=self.is_training,
                                           use_batch_renorm=use_batch_renorm,
                                           rmin=self.rmin, rmax=self.rmax, dmax=self.dmax
                                           )
                fan_in = fan_out

        # Now, the final layer outputs logits
        scope = 'logits'
        print(scope)
        with tf.variable_scope(scope):
            if preactivation:
                clip_values = {'rmin': self.rmin, 'rmax': self.rmax, 'dmax': self.dmax}
                inputs2next = tf.layers.batch_normalization(inputs=inputs2next, center=True, scale=True,
                                                            training=self.is_training, name='BatchNorm',
                                                            renorm=use_batch_renorm, renorm_clipping=clip_values
                                                            )
                #### With PReLU in logits block, it becomes another FC layer!!! Avoid this.
                # prelu_slope = tf.get_variable(name='prelu_slope',
                #                               shape=[1, fan_out],
                #                               dtype=tf.float32,
                #                               initializer=tf.truncated_normal_initializer(mean=0.25,
                #                                                                           stddev=tf.sqrt(2.0 / fan_out)
                #                                                                           ),
                #                               regularizer=None
                #                               )
                #
                # inputs2next = tf.add(tf.maximum(0., inputs2next), prelu_slope * tf.minimum(0., inputs2next),
                #                      name='activation_prelu'
                #                      )

            self.penultimate_features = inputs2next  # the penultimate layer's activations

            print('\t[%d, %d]' % (inputs2next.get_shape().as_list()[-1], self.num_classes))

            ### Variables ###
            weights = tf.get_variable(name='weights', shape=[inputs2next.get_shape().as_list()[-1], self.num_classes],
                                      dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      regularizer=tf.contrib.layers.l2_regularizer(scale=_lambda)
                                      )

            biases = tf.get_variable(name='biases', shape=[self.num_classes], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0),
                                     regularizer=None
                                     )

            ### Operations ###
            self.logits = tf.add(tf.matmul(inputs2next, weights), biases, name='logits')
            # self.predictions = tf.argmax(self.logits, axis=1, name='predictions')

            # This node may be changed to tf.nn.sigmoid or softmax depending on the Xentropy used for training.
            self.predictions_1hot = tf.nn.softmax(self.logits, name='predictions_1hot')

        scope = 'loss'
        with tf.name_scope(scope):
            Xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(self.labels_1hot,
                                                                                          name='labels_1hot_stopgrad'),
                                                                  logits=self.logits,
                                                                  name='Xentropy')
            #Xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_1hot,
            #                                                   logits=self.logits,
            #                                                   name='Xentropy')

            # epsilon = tf.constant(value=1e-14, name='epsilon')
            # Xentropy = self.labels_1hot * (-tf.log(self.predictions_1hot + epsilon)) * self.pos_weights

            # sample_weights = tf.reduce_sum(tf.multiply(self.labels_1hot, self.pos_weights), 1)
            # Xentropy = tf.losses.softmax_cross_entropy(onehot_labels=self.labels_1hot, logits=self.logits,
            # weights=sample_weights)

            self.loss = tf.reduce_mean(Xentropy, name='mean_Xentropy')

        scope = 'train_op'
        with tf.name_scope(scope):
            global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                            global_step=global_step,
                                                            decay_steps=decay_steps,
                                                            decay_rate=decay_rate,
                                                            staircase=True,
                                                            name='exp_lr_decay'
                                                            )
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='adam_optimizer')
            #optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum,
            #                                       use_nesterov=True,
            #                                       name='nesterov_momentum_optimizer')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, global_step=global_step)
                # self.train_op = optimizer.minimize(self.loss)

        scope = 'saver'
        with tf.name_scope(scope):
            self.saver = tf.train.Saver()

    ############################### End of build method ##############################################################

    def initialize(self):
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def train(self, tr_reader, max_iter=10000, batch_size=32, normalize=True, oversampling_threshold=0.5,
              val_step=1000, quick_dirty_val=False,
              val_source='/gpfs01/berens/user/mayhan/kaggle_dr_data/test_JF_BG_512/'):

        best_roc = 0.

        if max_iter <= 100:
            step = 1
        else:
            step = int(max_iter / 1000)
        self.diagnostics['losses'] = []
        self.diagnostics['avg_losses'] = []
        self.diagnostics['val_roc1'] = []
        self.diagnostics['val_roc2'] = []

        oversampling_threshold = int(max_iter * oversampling_threshold)

        total_loss = 0.

        print('Training %s ...' % self.name)
        for i in range(max_iter):
            if i < oversampling_threshold:
                sampling = 'balanced'
            else:
                sampling = 'stratified'

            feed_dict = self.feed_dict(reader=tr_reader, batch_size=batch_size, normalize=normalize, is_train=True,
                                       sampling=sampling, iter=i, max_iter=max_iter)

            _, loss_value = self.session.run([self.train_op, self.loss],
                                             feed_dict=feed_dict
                                             )
            total_loss = total_loss + loss_value
            avg_loss = total_loss/(i+1)
            self.diagnostics['losses'].append(loss_value)
            self.diagnostics['avg_losses'].append(avg_loss)

            if i % step == 0:
                # lr, momentum = self.session.run([self.learning_rate, self.momentum], feed_dict=feed_dict)
                lr = self.session.run(self.learning_rate, feed_dict=feed_dict)

                print("Iter %d/%d  Avg.Batch Loss: %f  Current Batch Loss: %f  Learn.rate : %g  %s" %
                      (i, max_iter, avg_loss, loss_value, lr, sampling))
                # print("Iter %d/%d  Avg.Batch Loss: %f  Current Batch Loss: %f  Learn.rate : %g  %s Momentum: %g" %
                #      (i, max_iter, avg_loss, loss_value, lr, sampling, momentum))
                # print("Iter %d/%d  Avg.Batch Loss: %f  Current Batch Loss: %f Sampling: %s" %
                #       (i, max_iter, avg_loss, loss_value, sampling))
                # print('Positive weights for cost-sensitive tackling of class imbalance: ' + str(pos_weights))

            if i % val_step == 0:
                print("Iter %d/%d, Validation once in %d steps..." % (i, max_iter, val_step))

                # It is fishy to create a new reader object everytime. In future, reader may have a reset method;
                # however, this is what I have for now!!! And it works :)
                val_reader = AdvancedReader(source=val_source,
                                            csv_file='/gpfs01/berens/user/mayhan/kaggle_dr_data/retinopathy_solution.csv',
                                            mode='val')
                _, _, roc_auc_val_onset1, roc_auc_val_onset2 = self.inference(val_reader, batch_size=batch_size,
                                                                              normalize=normalize,
                                                                              quick_dirty=quick_dirty_val)
                self.diagnostics['val_roc1'].append(roc_auc_val_onset1)
                self.diagnostics['val_roc2'].append(roc_auc_val_onset2)
                del val_reader

                if self.saver is not None and best_roc < roc_auc_val_onset1:
                    print('Current best : %g\t New best : %g' % (best_roc, roc_auc_val_onset1))
                    save_path = self.saver.save(self.session, self.model_path)
                    print("A better model found. Saving the model in path: %s" % save_path)
                    best_roc = roc_auc_val_onset1

        # Evaluate the validation performance for the last time
        val_reader = AdvancedReader(source=val_source,
                                    csv_file='/gpfs01/berens/user/mayhan/kaggle_dr_data/retinopathy_solution.csv',
                                    mode='val')
        _, _, roc_auc_val_onset1, roc_auc_val_onset2 = self.inference(val_reader, batch_size=batch_size,
                                                                      normalize=normalize,
                                                                      quick_dirty=quick_dirty_val)
        self.diagnostics['val_roc1'].append(roc_auc_val_onset1)
        self.diagnostics['val_roc2'].append(roc_auc_val_onset2)
        del val_reader

        if self.saver is not None and best_roc < roc_auc_val_onset1:
            print('Current best : %g\t New best : %g' % (best_roc, roc_auc_val_onset1))
            save_path = self.saver.save(self.session, self.model_path)
            print("A better model found. Saving the model in path: %s" % save_path)
            best_roc = roc_auc_val_onset1

        print('Average batch loss after %d iterations : %f' % (max_iter, avg_loss))
        print('Last batch loss after %d iterations : %f' % (max_iter, loss_value))

    def inference(self, reader, batch_size=32, normalize=True, quick_dirty=False):
        labels_1hot = []
        predictions_1hot = []
        i = 1
        progress = 0
        print('Evaluating %s ...' % self.name)
        while not reader.exhausted_test_cases:
            feed_dict = self.feed_dict(reader=reader, batch_size=batch_size, normalize=normalize, is_train=False,
                                       sampling='test', iter=1, max_iter=1)  # in test mode, sampling, iter, max_iter do not matter
            predictions, labels = self.session.run([self.predictions_1hot, self.labels_1hot], feed_dict=feed_dict)
            labels_1hot.append(labels)
            predictions_1hot.append(predictions)

            progress = progress + len(feed_dict[self.labels])
            #if i % 100 == 0:
            #    print("Progress : %d instances evaluated." % progress)
            i = i + 1
            if quick_dirty and i % 31 == 0:
                reader.exhausted_test_cases = True
        # print("Total instances processed %d." % progress)
        # penultimate_features = self.session.run(self.penultimate_features, feed_dict=feed_dict)
        # print('Shape of penultimate features (batch) : %s' % str(penultimate_features.shape))

        labels_1hot = np.squeeze([item for sublabels_1hot in labels_1hot for item in sublabels_1hot])
        predictions_1hot = np.squeeze([item for subpredictions_1hot in predictions_1hot for item in subpredictions_1hot])
        correct = np.equal(np.argmax(labels_1hot, axis=1), np.argmax(predictions_1hot, axis=1))
        acc = np.mean(np.asarray(correct, dtype=np.float32))
        print('Accuracy : %.5f' % acc)

        onset_level = 1
        labels_bin = np.greater_equal(np.argmax(labels_1hot, axis=1), onset_level)
        pred = np.sum(predictions_1hot[:, onset_level:], axis=1)
        fpr, tpr, _ = roc_curve(labels_bin, pred)
        roc_auc_onset1 = auc(fpr, tpr)
        print('Onset level = %d\t ROC-AUC: %.5f' % (onset_level, roc_auc_onset1))

        onset_level = 2
        labels_bin = np.greater_equal(np.argmax(labels_1hot, axis=1), onset_level)
        pred = np.sum(predictions_1hot[:, onset_level:], axis=1)
        fpr, tpr, _ = roc_curve(labels_bin, pred)
        roc_auc_onset2 = auc(fpr, tpr)
        print('Onset level = %d\t ROC-AUC: %.5f' % (onset_level, roc_auc_onset2))

        return labels_1hot, predictions_1hot, roc_auc_onset1, roc_auc_onset2

    def finalize(self):
        # Model saving features may be added here.
        #if self.saver is not None:
        #    # Save the variables to disk.
        #    save_path = self.saver.save(self.session, self.model_path)
        #    print("Model saved in path: %s" % save_path)

        self.session.close()
