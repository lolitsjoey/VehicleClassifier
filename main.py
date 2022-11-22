import munch as munch
import sklearn
from src.config import Config
from src.model_structure import classify_build_conv, derived_from_inception_model
from src.prepare_data import create_data_generators
import json
import tensorflow
import numpy as np

cfg = Config()


def get_truth_and_predictions(generator):
    y_pred = []
    y_true = []
    for batch_no in range(generator.__len__()):
        x, y = generator.__getitem__(batch_no)
        y_pred += [np.argmax(pred) for pred in model.predict(x)]
        y_true += [np.argmax(truth) for truth in y]
    return y_true, y_pred


def get_accuracy_precision_recall_f1(y_true, y_pred, dataset='train'):
    metrics['num_samples'] = len(y_true)
    metrics[f'{dataset}_accuracy'] = sklearn.metrics.accuracy_score(y_true, y_pred)
    metrics[f'{dataset}_precision'] = sklearn.metrics.precision_score(y_true, y_pred)
    metrics[f'{dataset}_recall'] = sklearn.metrics.recall_score(y_true, y_pred)
    metrics[f'{dataset}_f1score'] = sklearn.metrics.f1_score(y_true, y_pred)


def save_model_history_metrics(model, training_history, metrics, model_name):
    history_dict = training_history.history
    json.dump(history_dict, open(f'./saved_models/{model_name}_history.json', 'w'))
    json.dump(metrics, open(f'./saved_models/{model_name}_results.json', 'w'))


if __name__ == '__main__':
    train_generator, val_generator, test_generator = create_data_generators(cfg)

    custom_model = classify_build_conv()
    inceptionv3 = derived_from_inception_model()

    models_to_train = [custom_model]

    for model, model_name in zip(models_to_train, ['custom']):
        print(model.summary())
        if cfg.LOAD_ONLY:
            model = tensorflow.keras.models.load_model(f'./saved_models/{model_name}_weights.h5')
            history_dict = json.load(open(f'./saved_models/{model_name}_history.json', 'r'))
            training_history = munch.Munch(history=history_dict)

        else:
            training_history = model.fit(train_generator, validation_data=val_generator, shuffle=True, verbose=1,
                                         epochs=12, use_multiprocessing=False)
            model.save(f'./saved_models/{model_name}_weights.h5')

        train_generator, val_generator, test_generator = create_data_generators(cfg, augment=False)

        #train_true, train_pred = get_truth_and_predictions(train_generator)
        val_true, val_pred = get_truth_and_predictions(val_generator)
        test_true, test_pred = get_truth_and_predictions(test_generator)

        metrics = {}
        #get_accuracy_precision_recall_f1(train_true, train_pred, dataset='train')
        get_accuracy_precision_recall_f1(val_true, val_pred, dataset='val')
        get_accuracy_precision_recall_f1(test_true, test_pred, dataset='test')

        for key, val in metrics.items():
            print(f'{key} - {val}')

        if not cfg.LOAD_ONLY:
            save_model_history_metrics(model, training_history, metrics, model_name=model_name)



