import tensorflow as tf


def conv_prelu(inputs,
               kernel_shape=[3, 3], num_filters=32, strides=[1, 1], padding='SAME', reg_const=0.001,
               is_training=True, use_batch_renorm=False, rmin=0., rmax=float('Inf'), dmax=float('Inf')):

    print('\t[%d, %d], %d /%d' % (kernel_shape[0], kernel_shape[1], num_filters, strides[0]))

    weights = tf.get_variable(name='weights',
                              shape=[kernel_shape[0], kernel_shape[1], inputs.shape[-1], num_filters],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                          stddev=tf.sqrt(2.0 / (kernel_shape[0] *
                                                                                                kernel_shape[1] *
                                                                                                inputs.get_shape().as_list()[-1])
                                                                                         )
                                                                          ),
                              regularizer=tf.contrib.layers.l2_regularizer(scale=reg_const)
                              )

    biases = tf.get_variable(name='biases',
                             shape=[num_filters],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0),
                             regularizer=None
                             )

    preactivation = tf.add(tf.nn.conv2d(input=inputs,
                                        filter=weights,
                                        strides=[1, strides[0], strides[1], 1],
                                        padding=padding
                                        ),
                           biases,
                           name='preactivation'
                           )

    clip_values = {'rmin': rmin, 'rmax': rmax, 'dmax': dmax}
    preactivation = tf.layers.batch_normalization(inputs=preactivation, center=True, scale=True, training=is_training,
                                                  name='BatchNorm',
                                                  renorm=use_batch_renorm, renorm_clipping=clip_values
                                                  )

    prelu_slope = tf.get_variable(name='prelu_slope',
                                  shape=[1, 1, 1, num_filters],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0.25,
                                                                              stddev=tf.sqrt(2.0 / num_filters)
                                                                              ),
                                  regularizer=None
                                  )

    activation = tf.add(tf.maximum(0., preactivation), prelu_slope * tf.minimum(0., preactivation),
                        name='activation_prelu'
                        )

    return activation


def conv_prelu_preactivation(inputs,
                             kernel_shape=[3, 3], num_filters=32, strides=[1, 1], padding='SAME', reg_const=0.001,
                             is_training=True, use_batch_renorm=False, rmin=0., rmax=float('Inf'), dmax=float('Inf')):

    print('\t[%d, %d], %d /%d' % (kernel_shape[0], kernel_shape[1], num_filters, strides[0]))

    # clip_values = {'rmin': rmin, 'rmax': rmax, 'dmax': dmax}
    # preactivation = tf.layers.batch_normalization(inputs=inputs, center=True, scale=True, training=is_training,
    #                                               name='BatchNorm',
    #                                               renorm=use_batch_renorm, renorm_clipping=clip_values
    #                                               )
    #
    # prelu_slope = tf.get_variable(name='prelu_slope',
    #                               shape=[1, 1, 1, inputs.shape[-1]],
    #                               dtype=tf.float32,
    #                               initializer=tf.truncated_normal_initializer(mean=0.25,
    #                                                                           stddev=tf.sqrt(2.0 / num_filters)
    #                                                                           ),
    #                               regularizer=None
    #                               )
    #
    # activation = tf.add(tf.maximum(0., preactivation), prelu_slope * tf.minimum(0., preactivation),
    #                     name='activation_prelu'
    #                     )

    activation = inputs  # refer to inputs as activations and skip the BN and PReLU layers. Preactivated conv1 may be
    # the reason why networks starts learning too late, in comparison to the original block design.

    weights = tf.get_variable(name='weights',
                              shape=[kernel_shape[0], kernel_shape[1], activation.shape[-1], num_filters],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                          stddev=tf.sqrt(2.0 / (kernel_shape[0] *
                                                                                                kernel_shape[1] *
                                                                                                activation.get_shape().as_list()[-1])
                                                                                         )
                                                                          ),
                              regularizer=tf.contrib.layers.l2_regularizer(scale=reg_const)
                              )

    biases = tf.get_variable(name='biases',
                             shape=[num_filters],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0),
                             regularizer=None
                             )

    outputs = tf.add(tf.nn.conv2d(input=activation,
                                  filter=weights,
                                  strides=[1, strides[0], strides[1], 1],
                                  padding=padding
                                  ),
                     biases,
                     name='outputs'
                     )
    return outputs


def max_pool(inputs, kernel_shape=[2, 2], strides=[2, 2], padding='SAME'):
    return tf.nn.max_pool(inputs,
                          ksize=[1, kernel_shape[0], kernel_shape[1], 1],
                          strides=[1, strides[0], strides[1], 1],
                          padding=padding
                          )


def avg_pool(inputs, kernel_shape=[2, 2], strides=[2, 2], padding='SAME'):
    return tf.nn.avg_pool(inputs,
                          ksize=[1, kernel_shape[0], kernel_shape[1], 1],
                          strides=[1, strides[0], strides[1], 1],
                          padding=padding
                          )


def fc_prelu(inputs, fan_in, fan_out,
            reg_const=0.001, is_training=True, use_batch_renorm=False, rmin=0., rmax=float('Inf'), dmax=float('Inf')):

    print('\t[%d, %d]' % (fan_in, fan_out))

    weights = tf.get_variable(name='weights',
                              shape=[fan_in, fan_out],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                          stddev=tf.sqrt(2.0 / fan_in)
                                                                          ),
                              regularizer=tf.contrib.layers.l2_regularizer(scale=reg_const)
                              )

    biases = tf.get_variable(name='biases',
                             shape=[fan_out],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0),
                             regularizer=None
                             )

    preactivation = tf.add(tf.matmul(inputs, weights), biases, name='preactivation')

    clip_values = {'rmin': rmin, 'rmax': rmax, 'dmax': dmax}
    preactivation = tf.layers.batch_normalization(inputs=preactivation, center=True, scale=True, training=is_training,
                                                  name='BatchNorm',
                                                  renorm=use_batch_renorm, renorm_clipping=clip_values
                                                  )

    prelu_slope = tf.get_variable(name='prelu_slope',
                                  shape=[1, fan_out],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0.25,
                                                                              stddev=tf.sqrt(2.0 / fan_out)
                                                                              ),
                                  regularizer=None
                                  )

    activation = tf.add(tf.maximum(0., preactivation), prelu_slope * tf.minimum(0., preactivation),
                        name='activation_prelu'
                        )

    return activation


def fc_prelu_preactivation(inputs, fan_in, fan_out,
                           reg_const=0.001, is_training=True, use_batch_renorm=False, rmin=0., rmax=float('Inf'), dmax=float('Inf')):

    print('\t[%d, %d]' % (fan_in, fan_out))

    clip_values = {'rmin': rmin, 'rmax': rmax, 'dmax': dmax}
    preactivation = tf.layers.batch_normalization(inputs=inputs, center=True, scale=True, training=is_training,
                                                  name='BatchNorm',
                                                  renorm=use_batch_renorm, renorm_clipping=clip_values
                                                  )

    prelu_slope = tf.get_variable(name='prelu_slope',
                                  shape=[1, fan_in],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0.25,
                                                                              stddev=tf.sqrt(2.0 / fan_in)
                                                                              ),
                                  regularizer=None
                                  )

    activation = tf.add(tf.maximum(0., preactivation), prelu_slope * tf.minimum(0., preactivation),
                        name='activation_prelu'
                        )

    weights = tf.get_variable(name='weights',
                              shape=[fan_in, fan_out],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                          stddev=tf.sqrt(2.0 / fan_in)
                                                                          ),
                              regularizer=tf.contrib.layers.l2_regularizer(scale=reg_const)
                              )

    biases = tf.get_variable(name='biases',
                             shape=[fan_out],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0),
                             regularizer=None
                             )

    return tf.add(tf.matmul(activation, weights), biases, name='outputs')


## Residual blocks for ResNets

#  Bottleneck design for residual blocks

def resnet_bottleneck_identity_block(inputs,
                                     kernel_shapes=[[1, 1], [3, 3], [1, 1]],
                                     num_filters=[64, 64, 256], strides=[1, 1], padding='SAME', reg_const=0.001,
                                     is_training=True, use_batch_renorm=False,
                                     rmin=0., rmax=float('Inf'), dmax=float('Inf')
                                     ):
    assert len(kernel_shapes) == len(num_filters), "Number of kernel shapes does not match the number of filters"

    input2next = inputs  # input2next iterates over the weight layers in residual block
    i = 1
    for kernel_shape, filters in zip(kernel_shapes, num_filters):
        print('\t[%d, %d], %d /%d' % (kernel_shape[0], kernel_shape[1], filters, strides[0]))

        weights = tf.get_variable(name='weights_' + str(i),
                                  shape=[kernel_shape[0], kernel_shape[1], input2next.shape[-1], filters],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                              stddev=tf.sqrt(2.0 / (kernel_shape[0] *
                                                                                                    kernel_shape[1] *
                                                                                                    input2next.get_shape().as_list()[-1])
                                                                                             )
                                                                              ),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=reg_const)
                                  )
        biases = tf.get_variable(name='biases_' + str(i),
                                 shape=[filters],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0),
                                 regularizer=None
                                 )
        preactivation = tf.add(tf.nn.conv2d(input=input2next, filter=weights, strides=[1, strides[0], strides[1], 1],
                                            padding=padding
                                            ),
                               biases,
                               name='preactivation_' + str(i)
                               )

        clip_values = {'rmin': rmin, 'rmax': rmax, 'dmax': dmax}
        preactivation = tf.layers.batch_normalization(inputs=preactivation, center=True, scale=True, training=is_training,
                                                      name='BatchNorm_' + str(i),
                                                      renorm=use_batch_renorm, renorm_clipping=clip_values
                                                      )
        # activation = tf.nn.relu(preactivation, name='activation_relu_' + str(i))

        if i == len(kernel_shapes): # if last, save the preactivation for later
            input2next = preactivation
        else:
            input2next = tf.nn.relu(preactivation, name='activation_relu_' + str(i))

        i = i + 1

    # Now, add the residual to the input from the clean path
    return tf.nn.relu(tf.add(inputs, input2next, name='identity_plus_residual'), name='identity_block_activation')


def resnet_bottleneck_head_block(inputs,
                                 kernel_shapes=[[1, 1], [3, 3], [1, 1]],
                                 num_filters=[64, 64, 256], strides=[2, 2], padding='SAME', reg_const=0.001,
                                 is_training=True, use_batch_renorm=False,
                                 rmin=0., rmax=float('Inf'), dmax=float('Inf')
                                 ):
    assert len(kernel_shapes) == len(num_filters), "Number of kernel shapes does not match the number of filters"

    if strides[0] > 1 or strides[1] > 1:
        downsampling_head = True
    else:
        downsampling_head = False

    input2next = inputs  # input2next iterates over the weight layers in residual block
    i = 1
    for kernel_shape, filters in zip(kernel_shapes, num_filters):
        print('\t[%d, %d], %d /%d' % (kernel_shape[0], kernel_shape[1], filters, strides[0]))

        weights = tf.get_variable(name='weights_' + str(i),
                                  shape=[kernel_shape[0], kernel_shape[1], input2next.shape[-1], filters],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                              stddev=tf.sqrt(2.0 / (kernel_shape[0] *
                                                                                                    kernel_shape[1] *
                                                                                                    input2next.get_shape().as_list()[-1])
                                                                                             )
                                                                              ),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=reg_const)
                                  )
        biases = tf.get_variable(name='biases_' + str(i),
                                 shape=[filters],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0),
                                 regularizer=None
                                 )
        preactivation = tf.add(tf.nn.conv2d(input=input2next, filter=weights, strides=[1, strides[0], strides[1], 1],
                                            padding=padding
                                            ),
                               biases,
                               name='preactivation_' + str(i)
                               )

        clip_values = {'rmin': rmin, 'rmax': rmax, 'dmax': dmax}
        preactivation = tf.layers.batch_normalization(inputs=preactivation, center=True, scale=True, training=is_training,
                                                      name='BatchNorm_' + str(i),
                                                      renorm=use_batch_renorm, renorm_clipping=clip_values
                                                      )
        # activation = tf.nn.relu(preactivation, name='activation_relu_' + str(i))

        if i == len(kernel_shapes): # if last, save the activation for later
            input2next = preactivation
        else:
            input2next = tf.nn.relu(preactivation, name='activation_relu_' + str(i))

        i = i + 1
        strides = [1, 1]  # After downsampling via the first conv. layer, no more downsampling.

    # # # # # # # # # # # # #
    # Clean path operations #
    # # # # # # # # # # # # #

    # =============== Downsampling via max pooling + zero padding to match dimensions ==================
    # Downsample the inputs from clean path, and pad zeros to match dimension
    # Note to self about the usage of strides: if a head block is not downsampling at the top,
    # then, it should not downsample the inputs from clean path, either.
    # inputs_halved = max_pool(inputs, kernel_shape=strides, strides=strides, padding='SAME')

    if downsampling_head:
        inputs_halved = max_pool(inputs, kernel_shape=[2, 2], strides=[2, 2], padding='SAME')
    else:
        inputs_halved = inputs  # This is for the first head block that does not downsample.

    paddings = tf.constant([[0, 0], [0, 0], [0, 0],
                            [int((input2next.get_shape().as_list()[-1] - inputs_halved.get_shape().as_list()[-1]) / 2),
                             int((input2next.get_shape().as_list()[-1] - inputs_halved.get_shape().as_list()[-1]) / 2)]],
                           dtype=tf.int32)
    inputs_padded = tf.pad(tensor=inputs_halved, paddings=paddings, mode='CONSTANT', name='zero_padding',
                           constant_values=0)

    return tf.nn.relu(tf.add(inputs_padded, input2next, name='downsampled_padded_plus_residual'), name='identity_block_activation')

    # =============== Downsampling via 1x1 conv. Zero padding not required. conv with the correct number of filters will
    # take care of matching the dimensions ==================

    # if downsampling_head:
    #     strides = [2, 2]
    #
    # else:
    #     strides = [1, 1]
    #
    # kernel_shape = [1, 1]
    #
    # weights = tf.get_variable(name='weights_' + 'head_1x1',
    #                           shape=[kernel_shape[0], kernel_shape[1], inputs.shape[-1], num_filters[-1]],
    #                           dtype=tf.float32,
    #                           initializer=tf.truncated_normal_initializer(mean=0.0,
    #                                                                       stddev=tf.sqrt(2.0 / (kernel_shape[0] *
    #                                                                                             kernel_shape[1] *
    #                                                                                             inputs.get_shape().as_list()[-1])
    #                                                                                      )
    #                                                                       ),
    #                           regularizer=tf.contrib.layers.l2_regularizer(scale=reg_const)
    #                           )
    # biases = tf.get_variable(name='biases_' + 'head_1x1',
    #                          shape=[num_filters[-1]],
    #                          dtype=tf.float32,
    #                          initializer=tf.constant_initializer(0.0),
    #                          regularizer=None
    #                          )
    # inputs_halved = tf.add(tf.nn.conv2d(input=inputs, filter=weights, strides=[1, strides[0], strides[1], 1],
    #                                     padding=padding),
    #                        biases,
    #                        name='inputs_halved' + 'head_1x1'
    #                        )
    #
    # clip_values = {'rmin': rmin, 'rmax': rmax, 'dmax': dmax}
    # inputs_halved = tf.layers.batch_normalization(inputs=inputs_halved, center=True, scale=True,
    #                                               training=is_training,
    #                                               name='BatchNorm_' + 'head_1x1',
    #                                               renorm=use_batch_renorm, renorm_clipping=clip_values
    #                                               )
    #
    # # Now, add the residual to the input from the clean path
    # return tf.nn.relu(tf.add(inputs_halved, input2next, name='downsampled_plus_residual'),
    #                   name='identity_block_activation')


##############################################
## Pre-Activation

def resnet_bottleneck_identity_block_preactivation(inputs,
                                                   kernel_shapes=[[1, 1], [3, 3], [1, 1]],
                                                   num_filters=[64, 64, 256], strides=[1, 1], padding='SAME', reg_const=0.001,
                                                   is_training=True, use_batch_renorm=False,
                                                   rmin=0., rmax=float('Inf'), dmax=float('Inf')
                                                   ):
    assert len(kernel_shapes) == len(num_filters), "Number of kernel shapes does not match the number of filters"

    input2next = inputs  # input2next iterates over the weight layers in residual block
    i = 1
    for kernel_shape, filters in zip(kernel_shapes, num_filters):
        print('\t[%d, %d], %d /%d' % (kernel_shape[0], kernel_shape[1], filters, strides[0]))

        clip_values = {'rmin': rmin, 'rmax': rmax, 'dmax': dmax}
        preactivation = tf.layers.batch_normalization(inputs=input2next, center=True, scale=True, training=is_training,
                                                      name='BatchNorm_' + str(i),
                                                      renorm=use_batch_renorm, renorm_clipping=clip_values
                                                      )
        activation = tf.nn.relu(preactivation, name='activation_relu_' + str(i))

        weights = tf.get_variable(name='weights_' + str(i),
                                  shape=[kernel_shape[0], kernel_shape[1], activation.shape[-1], filters],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                              stddev=tf.sqrt(2.0 / (kernel_shape[0] *
                                                                                                    kernel_shape[1] *
                                                                                                    activation.get_shape().as_list()[-1])
                                                                                             )
                                                                              ),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=reg_const)
                                  )
        biases = tf.get_variable(name='biases_' + str(i),
                                 shape=[filters],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0),
                                 regularizer=None
                                 )
        input2next = tf.add(tf.nn.conv2d(input=activation, filter=weights, strides=[1, strides[0], strides[1], 1],
                                         padding=padding
                                         ),
                            biases,
                            name='conv_out_' + str(i)
                            )
        i = i + 1

    # Now, add the residual to the input from the clean path
    return tf.add(inputs, input2next, name='identity_plus_residual')


def resnet_bottleneck_head_block_preactivation(inputs,
                                               kernel_shapes=[[1, 1], [3, 3], [1, 1]],
                                               num_filters=[64, 64, 256], strides=[2, 2], padding='SAME', reg_const=0.001,
                                               is_training=True, use_batch_renorm=False,
                                               rmin=0., rmax=float('Inf'), dmax=float('Inf')
                                               ):
    assert len(kernel_shapes) == len(num_filters), "Number of kernel shapes does not match the number of filters"

    if strides[0] > 1 or strides[1] > 1:
        downsampling_head = True
    else:
        downsampling_head = False

    input2next = inputs  # input2next iterates over the weight layers in residual block
    i = 1
    for kernel_shape, filters in zip(kernel_shapes, num_filters):
        print('\t[%d, %d], %d /%d' % (kernel_shape[0], kernel_shape[1], filters, strides[0]))

        clip_values = {'rmin': rmin, 'rmax': rmax, 'dmax': dmax}
        preactivation = tf.layers.batch_normalization(inputs=input2next, center=True, scale=True, training=is_training,
                                                      name='BatchNorm_' + str(i),
                                                      renorm=use_batch_renorm, renorm_clipping=clip_values
                                                      )
        activation = tf.nn.relu(preactivation, name='activation_relu_' + str(i))

        weights = tf.get_variable(name='weights_' + str(i),
                                  shape=[kernel_shape[0], kernel_shape[1], activation.shape[-1], filters],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                              stddev=tf.sqrt(2.0 / (kernel_shape[0] *
                                                                                                    kernel_shape[1] *
                                                                                                    activation.get_shape().as_list()[-1])
                                                                                             )
                                                                              ),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=reg_const)
                                  )
        biases = tf.get_variable(name='biases_' + str(i),
                                 shape=[filters],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0),
                                 regularizer=None
                                 )
        input2next = tf.add(tf.nn.conv2d(input=activation, filter=weights, strides=[1, strides[0], strides[1], 1],
                                         padding=padding
                                         ),
                            biases,
                            name='conv_out_' + str(i)
                            )
        i = i + 1
        strides = [1, 1]  # After downsampling via the first conv. layer, no more downsampling.

    # # # # # # # # # # # # #
    # Clean path operations #
    # # # # # # # # # # # # #

    # =============== Downsampling via max pooling + zero padding to match dimensions ==================
    # Downsample the inputs from clean path, and pad zeros to match dimension
    # Note to self about the usage of strides: if a head block is not downsampling at the top,
    # then, it should not downsample the inputs from clean path, either.
    # inputs_halved = max_pool(inputs, kernel_shape=strides, strides=strides, padding='SAME')

    if downsampling_head:
        inputs_halved = max_pool(inputs, kernel_shape=[2, 2], strides=[2, 2], padding='SAME')
    else:
        inputs_halved = inputs  # This is for the first head block that does not downsample.

    paddings = tf.constant([[0, 0], [0, 0], [0, 0],
                            [int((input2next.get_shape().as_list()[-1] - inputs_halved.get_shape().as_list()[-1]) / 2),
                             int((input2next.get_shape().as_list()[-1] - inputs_halved.get_shape().as_list()[-1]) / 2)]],
                           dtype=tf.int32)
    inputs_padded = tf.pad(tensor=inputs_halved, paddings=paddings, mode='CONSTANT', name='zero_padding',
                           constant_values=0)

    # Now, add the residual to the input from the clean path
    return tf.add(inputs_padded, input2next, name='downsampled_padded_plus_residual')

    # =============== Downsampling via 1x1 conv. Zero padding not required. conv with the correct number of filters will
    # take care of matching the dimensions ==================

    # if downsampling_head:
    #     strides = [2, 2]
    #
    # else:
    #     strides = [1, 1]
    #
    # kernel_shape = [1, 1]
    #
    # clip_values = {'rmin': rmin, 'rmax': rmax, 'dmax': dmax}
    # preactivation = tf.layers.batch_normalization(inputs=inputs, center=True, scale=True, training=is_training,
    #                                               name='BatchNorm_' + 'head_1x1',
    #                                               renorm=use_batch_renorm, renorm_clipping=clip_values
    #                                               )
    # activation = tf.nn.relu(preactivation, name='activation_relu_' + 'head_1x1')
    #
    # weights = tf.get_variable(name='weights_' + 'head_1x1',
    #                           shape=[kernel_shape[0], kernel_shape[1], activation.shape[-1], num_filters[-1]],
    #                           dtype=tf.float32,
    #                           initializer=tf.truncated_normal_initializer(mean=0.0,
    #                                                                       stddev=tf.sqrt(2.0 / (kernel_shape[0] *
    #                                                                                             kernel_shape[1] *
    #                                                                                             activation.get_shape().as_list()[-1])
    #                                                                                      )
    #                                                                       ),
    #                           regularizer=tf.contrib.layers.l2_regularizer(scale=reg_const)
    #                           )
    # biases = tf.get_variable(name='biases_' + 'head_1x1',
    #                          shape=[num_filters[-1]],
    #                          dtype=tf.float32,
    #                          initializer=tf.constant_initializer(0.0),
    #                          regularizer=None
    #                          )
    # inputs_halved = tf.add(tf.nn.conv2d(input=activation, filter=weights, strides=[1, strides[0], strides[1], 1],
    #                                     padding=padding
    #                                     ),
    #                        biases,
    #                        name='conv_out_' + 'head_1x1'
    #                        )
    #
    # # Now, add the residual to the input from the clean path
    # return tf.add(inputs_halved, input2next, name='downsampled_plus_residual')
