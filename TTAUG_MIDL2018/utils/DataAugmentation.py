import tensorflow as tf

# @staticmethod


def do_nothing(inputs):
    return inputs


def crop_and_resize(image):
    # A random box for cropping is generated per input image
    box_margin = 0.33
    random_box = tf.transpose(
        [tf.random_uniform(shape=[1], minval=0., maxval=box_margin, dtype=tf.float32, name='randombox_y1'),
         tf.random_uniform(shape=[1], minval=0., maxval=box_margin, dtype=tf.float32, name='randombox_x1'),
         tf.random_uniform(shape=[1], minval=1.-box_margin, maxval=1., dtype=tf.float32, name='randombox_y2'),
         tf.random_uniform(shape=[1], minval=1.-box_margin, maxval=1., dtype=tf.float32, name='randombox_x2')
         ]
    )

    image = tf.expand_dims(image, 0)  # tf.image.crop_and_resize() expects 4-D tensor. Thus, add an extra dimension
    image = tf.cond(
        tf.squeeze(tf.greater_equal(tf.random_uniform(shape=[1], minval=0., maxval=1.0, dtype=tf.float32,
                                                      name='coin_toss'),
                                    tf.constant(value=0.5, dtype=tf.float32, name='coin_threshold')
                                    )
                   ),
        true_fn=lambda: tf.image.crop_and_resize(image=image,
                                                 boxes=random_box,
                                                 box_ind=tf.range(0, image.get_shape().as_list()[0]),
                                                 crop_size=[image.get_shape().as_list()[1], image.get_shape().as_list()[2]],
                                                 method='bilinear',
                                                 extrapolation_value=0,
                                                 name='crop_and_resize'
                                                 ),
        false_fn=lambda: do_nothing(image)
    )
    image = tf.squeeze(image)  # Remove the extra dimension added for tf.image.crop_and_resize()

    return image


def data_augmentation(inputs):
    scope = 'data_aug'
    with tf.variable_scope(scope):
        #  ############  data augmentation on the fly  ###################
        #  PART 0: First, randomly crop images. Randomness has two folds:
        #  i) Coin toss: To crop or not to crop ii) Bounding box corners are randomly generated
        #  This part can considered as an extension to the sampling process.
        #  random crops from images resized to original shape (zooming effect)
        inputs = tf.map_fn(lambda img: crop_and_resize(img), inputs, dtype=tf.float32, name='random_crop_resize')

        # PART 1: Manipulation of pixels values via brightness, hue, saturation, and contrast adjustments

        # random brightness adjustment with delta sampled from [-max_delta, max_delta]
        inputs = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=0.5), inputs,
                           dtype=tf.float32, name='random_brightness')

        # random hue adjustments with delta sampled from [-max_delta, max_delta]. max_delta must be in [0, 0.5].
        inputs = tf.map_fn(lambda img: tf.image.random_hue(img, max_delta=0.5), inputs, dtype=tf.float32,
                           name='random_hue')

        # random saturation adjustments: i) lower >= 0 and ii) lower < upper
        inputs = tf.map_fn(lambda img: tf.image.random_saturation(img, lower=0., upper=3.), inputs,
                           dtype=tf.float32, name='random_saturation')

        # random contrast adjustments: i) lower >= 0 and ii) lower < upper
        inputs = tf.image.random_contrast(inputs, lower=0., upper=3.)

        # make sure that pixel values are in [0., 1.]
        inputs = tf.minimum(inputs, 1.0)
        inputs = tf.maximum(inputs, 0.0)

        # PART 2: Physical transformations on images: Flip LR, Flip UD, Rotate

        # randomly mirror images horizontally
        inputs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), inputs, dtype=tf.float32,
                           name='random_flip_lr')

        # randomly mirror images vertically
        inputs = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), inputs, dtype=tf.float32,
                           name='random_flip_ud')

        # random translations
        #inputs = tf.contrib.image.translate(inputs,
        #                                    translations=tf.random_uniform(shape=[tf.shape(inputs)[0], 2],
        #                                                                   minval=-50, maxval=50, dtype=tf.float32
        #                                                                   ),
        #                                    interpolation='NEAREST',
        #                                    name=None
        #                                    )


        # random rotations
        inputs = tf.contrib.image.rotate(inputs,
                                         angles=tf.random_uniform(shape=[tf.shape(inputs)[0]],  # this has to inferred shape; get_shape() returns None.
                                                                  minval=0, maxval=360, dtype=tf.float32
                                                                  ),
                                         interpolation='NEAREST'
                                         )
    return inputs


def data_augmentation_wrapper(inputs):

    inputs = tf.cond(
        tf.squeeze(tf.greater_equal(tf.random_uniform(shape=[1], minval=0., maxval=1.0, dtype=tf.float32,
                                                      name='coin_toss'),
                                    tf.constant(value=0.5, dtype=tf.float32, name='data_augmentation_threshold')
                                    )
                   ),
        true_fn=lambda: data_augmentation(inputs),
        false_fn=lambda: do_nothing(inputs)
    )

    return inputs
