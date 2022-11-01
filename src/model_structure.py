import tensorflow.keras.optimizers
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.applications.inception_v3 import InceptionV3

from src.config import Config

cfg = Config()


def classify_build_conv():
    input = Input(shape=(cfg.im_dim, cfg.im_dim, 3))
    conv_1 = Conv2D(16, 15, 15, activation='relu')(input)
    pool_1 = MaxPooling2D()(conv_1)
    norm_1 = BatchNormalization()(pool_1)

    flat = Flatten()(norm_1)
    dense_1 = Dense(128, activation='relu')(flat)
    dense_2 = Dense(32, activation='sigmoid')(dense_1)
    output = Dense(1, activation='sigmoid')(dense_2)

    model = Model(inputs=input, outputs=output)
    model.compile(loss=tensorflow.keras.losses.BinaryCrossentropy(),
                  optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.00001),
                  metrics=['accuracy'])
    return model


def derived_from_inception_model():
    base_model = InceptionV3(weights='imagenet', include_top=True)
    output1 = Dense(256, activation='relu')(base_model.output)
    output2 = Dense(32, activation='relu')(output1)
    output4 = Dense(1, activation='sigmoid')(output2)
    model = Model(inputs=base_model.input, outputs=output4, name='vehicle_classifier')
    model.compile(loss=tensorflow.keras.losses.BinaryCrossentropy(),
                  optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=['accuracy'])
    return model
