from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_addons as tfa


def facenet(input_shape):
    # create inceptions resnet v2
    net = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        input_shape=input_shape,
    )

    # frozen layers
    for layer in net.layers:
        layer.trainable = False

    x = net.output
    x = tf.keras.layers.Flatten()(x)
    # No activation on final dense layer
    x = tf.keras.layers.Dense(1024, activation=None)(x)
    # L2 normalize embeddings
    out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)

    model = tf.keras.models.Model(inputs=net.inputs, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss(),
    )

    return model
