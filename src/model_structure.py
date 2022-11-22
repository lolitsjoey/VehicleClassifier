import tensorflow.keras.optimizers
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import L1L2

from src.config import Config

cfg = Config()


def classify_build_conv():
    input = Input(shape=(cfg.im_dim, cfg.im_dim, 3))
    conv_1 = Conv2D(8, 4, 1, activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5))(input)
    pool_1 = MaxPooling2D(padding='same')(conv_1)
    norm_1 = BatchNormalization()(pool_1)

    conv_2 = Conv2D(16, 2, 1, activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5))(norm_1)
    pool_2 = MaxPooling2D(padding='same')(conv_2)
    norm_2 = BatchNormalization()(pool_2)
    #
    # flat_1 = Flatten()(norm_1)
    # flat_2 = Flatten()(conv_2)
    # flat_3 = Flatten()(norm_2)

    resize_1 = tensorflow.keras.layers.Resizing(norm_1.shape[1], norm_1.shape[2])(norm_2)
    resize_2 = tensorflow.keras.layers.Resizing(norm_1.shape[1], norm_1.shape[2])(conv_2)

    concat_1 = concatenate([resize_1, resize_2, norm_1])

    conv_3 = Conv2D(4, 4, 1, activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5))(concat_1)
    pool_3 = MaxPooling2D(padding='same')(conv_3)
    norm_3 = BatchNormalization()(pool_3)

    flat_1 = Flatten()(norm_2)
    flat_2 = Flatten()(norm_3)

    dense_1 = Dense(32, activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5))(flat_1)
    dense_2 = Dense(32, activation='relu', kernel_regularizer=L1L2(l1=1e-5, l2=1e-5))(flat_2)

    concat_2 = concatenate([dense_1, dense_2])

    dense_2 = Dense(8, activation='relu')(concat_2)
    output = Dense(2, activation='softmax')(dense_2)

    model = Model(inputs=input, outputs=output)
    model.compile(loss=CategoricalCrossentropy(),
                  optimizer=Adam(learning_rate=0.000001),
                  metrics=['accuracy', Recall(), Precision()])
    return model


def derived_from_inception_model():
    base_model = InceptionV3(weights='imagenet', include_top=False)
    output1 = Dense(256, activation='relu')(base_model.output)
    output2 = Dense(32, activation='relu')(output1)
    output4 = Dense(2, activation='softmax')(output2)
    model = Model(inputs=base_model.input, outputs=output4, name='vehicle_classifier')
    model.compile(loss=CategoricalCrossentropy(),
                  optimizer=Adam(learning_rate=0.01),
                  metrics=['accuracy'])
    return model
