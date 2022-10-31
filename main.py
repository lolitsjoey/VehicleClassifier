import matplotlib.pyplot as plt
from keras import Model, Input

from src.config import Config
from src.prepare_data import create_data_generators
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense
import tensorflow as tf
import keras
import numpy as np

cfg = Config()

if __name__ == '__main__':
    train_generator, val_generator, test_generator = create_data_generators(cfg)

    base_model = InceptionV3(weights='imagenet', include_top=True)
    output1 = Dense(512, activation='relu')(base_model.output)
    # output1 = Dense(128, activation='relu')(output1)
    # output1 = Dense(32, activation='relu')(output1)
    output4 = Dense(2, activation='softmax')(output1)

    model = Model(inputs=base_model.input, outputs=output4, name='vehicle_classifier')
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['categorical_accuracy'])
    training_history = model.fit(train_generator, validation_data=val_generator, shuffle=True, verbose=1, epochs=18)
    model.save('./saved_models/model_new_aug.h5')
    #model = keras.models.load_model('./saved_models/model.h5')
    #model.evaluate(test_generator)

    train_generator, val_generator, test_generator = create_data_generators(cfg, augment=False)

    y_pred = []
    y_true = []
    for batch_no in range(test_generator.__len__()):
        x, y = test_generator.__getitem__(batch_no)
        y_pred += list(np.argmax(model.predict(x), axis=1))
        print(np.unique(y_pred, return_counts=True))
        y_true += list(np.argmax(y, axis=1))

        for idx, image in enumerate(x):
            if np.random.choice(100) == 5:
                plt.imshow(image)
                plt.title(f'Predicted {y_pred[idx]} -  {y_true[idx]} Actual')
                plt.show()

    print(f'Test Accuracy: {sum(np.array(y_pred) == np.array(y_true))/len(y_true)}')

    y_pred = []
    y_true = []
    for batch_no in range(train_generator.__len__()):
        x, y = train_generator.__getitem__(batch_no)
        y_pred += list(np.argmax(model.predict(x), axis=1))
        y_true += list(np.argmax(y, axis=1))
        print(np.unique(y_pred, return_counts=True))

        for idx, image in enumerate(x):
            if np.random.choice(100) == 5:
                plt.imshow(image)
                plt.title(f'Predicted {y_pred[idx]} -  {y_true[idx]} Actual')
                plt.show()

    print(f'Train Accuracy: {sum(np.array(y_pred) == np.array(y_true))/len(y_true)}')

    y_pred = []
    y_true = []
    for batch_no in range(val_generator.__len__()):
        x, y = val_generator.__getitem__(batch_no)
        y_pred += list(np.argmax(model.predict(x), axis=1))
        y_true += list(np.argmax(y, axis=1))
        print(np.unique(y_pred, return_counts=True))

        for idx, image in enumerate(x):
            if np.random.choice(100) == 5:
                plt.imshow(image)
                plt.title(f'Predicted {y_pred[idx]} -  {y_true[idx]} Actual')
                plt.show()

    print(f'Val Accuracy: {sum(np.array(y_pred) == np.array(y_true)) / len(y_true)}')

    import json

    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = training_history.history
    # Save it under the form of a json file
    json.dump(history_dict, open('./saved_models/history.json', 'w'))