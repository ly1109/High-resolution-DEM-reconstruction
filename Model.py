import tensorflow as tf
from keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Activation, GlobalMaxPool2D, Reshape, Dense, \
    Multiply, Add, Conv2DTranspose
from keras.models import Model

from Datasets import dataset

rows = 64
cols = 64


def Generator_L2H():
    inputs = Input((rows, cols, 1))

    x1 = Conv2D(128, 2, activation=tf.nn.leaky_relu, padding='same')(inputs)
    x2 = Conv2D(128, 2, activation=tf.nn.leaky_relu, padding='same')(x1)
    x3 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')(x2)
    x4 = Conv2D(128, 3, activation=tf.nn.leaky_relu, padding='same')(x3)
    x5 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(x4)
    x6 = Conv2DTranspose(128, 2, padding='same')(x5)
    x6 = Add()([x4, x6])
    x7 = Conv2DTranspose(128, 2, padding='same')(x6)
    x8 = Conv2DTranspose(128, 3, padding='same')(x7)
    x8 = Add()([x2, x8])
    x9 = Conv2DTranspose(64, 3, padding='same')(x8)
    x10 = Conv2DTranspose(1, 3, padding='same')(x9)
    x10 = Add()([inputs, x10])

    conv1 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(x10)
    conv2 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv1)
    conv3 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv2)
    conv4 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv3)
    conv5 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv4)
    conv6 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv5)
    conv7 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv6)
    conv8 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv7)
    conv9 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv8)
    conv10 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv9)
    conv11 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv10)
    conv12 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv11)

    concat1 = Concatenate()([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12])

    x11 = GlobalMaxPool2D()(concat1)
    x11 = Reshape((1, 1, 64 * 12))(x11)
    x11 = Dense(64 * 12)(x11)
    x11 = Activation(tf.nn.relu)(x11)
    x11 = Dense(64 * 12)(x11)
    x11 = Activation(tf.nn.sigmoid)(x11)
    x11 = Multiply()([x11, concat1])

    conv13 = Conv2D(64, 1, activation=tf.nn.leaky_relu, padding='same')(x11)

    conv14 = Conv2D(64, 1, activation=tf.nn.leaky_relu, padding='same')(x11)
    conv14 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(conv14)

    concat2 = Concatenate()([conv13, conv14])

    conv15 = Conv2D(144, 3, activation=tf.nn.leaky_relu, padding='same')(concat2)

    pix16 = tf.nn.depth_to_space(conv15, 6)

    conv16 = Conv2D(64, 3, activation=tf.nn.leaky_relu, padding='same')(pix16)

    up17 = UpSampling2D(size=(6, 6), interpolation='lanczos5')(inputs)

    add1 = Add()([up17, conv16])

    conv17 = Conv2D(1, 1, activation='linear')(add1)

    model = Model(inputs=inputs, outputs=conv17)

    return model


model = Generator_L2H()
model.summary()

x_train, y_train = dataset()


lr_schedule = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-4,
                                                       decay_steps=2250,
                                                       decay_rate=0.50,
                                                       staircase=True)

optimizer = tf.optimizers.Adam(learning_rate=lr_schedule,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False,
                               clipvalue=0.5,
                               clipnorm=1)


def loss_mse(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true-y_pred))
    return 10 * loss


model.compile(optimizer=optimizer, loss=[loss_mse], metrics=['mae'])  # "accuracy"r_squared, PSNRLoss

history = model.fit(x_train, y_train, batch_size=16, epochs=100, shuffle=True, verbose=True)

