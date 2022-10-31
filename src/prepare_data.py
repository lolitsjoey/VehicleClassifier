import os
import shutil
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image

from src.augmentation import make_transform
from src.config import Config
cfg = Config()


def remove_unnecessary_columns(df, unecessary_columns, inplace=True):
    if not inplace:
        df = df.copy()

    for col_name in unecessary_columns:
        if col_name in df.columns: del df[col_name]

    if not inplace:
        return df


def _image_has_multiple_labels(metadata):
    return len(np.unique(metadata['truth_label'])) != 1


class DataDownloader:
    def __init__(self, cfg):
        self.config = cfg
        
        if cfg.REDOWNLOAD:
            if os.path.exists(os.path.split(cfg.extract_path)[0]):
                shutil.rmtree(os.path.split(cfg.extract_path)[0])
            os.makedirs(os.path.split(cfg.extract_path)[0])
            self.download_and_extract()

        data = self.tidy_data()
        data = data.groupby('truth_label').apply(lambda x: x.sample(data.truth_label.value_counts().min()))
        data = data.reset_index(drop=True)

        train_and_test, val = train_test_split(data, test_size=0.1)
        train, test = train_test_split(train_and_test, test_size=1/9)
        self.train, self.val, self.test = train, val, test

    def multiply_frame_by_aug_fac(self, df):
        df = pd.concat([df] * self.config.augmentation_fac)
        df = df.reset_index(drop=True)
        return df

    def download_and_extract(self):
        url_response = urlopen(self.config.data_url)

        zip_file_path = f'{self.config.extract_path}_compressed.zip'

        with open(zip_file_path, 'wb') as f:
            f.write(url_response.read())
        
        zf = ZipFile(zip_file_path)
        zf.extractall(path=f'{self.config.extract_path}/')
        zf.close()

    def _image_exists(self, lab):
        return os.path.isfile(f'{self.config.extract_path}/Images/{lab}')

    def tidy_data(self):
        train_labels = pd.read_csv(f'{self.config.extract_path}/Labels/CSV Format/train_labels.csv')
        test_labels  = pd.read_csv(f'{self.config.extract_path}/Labels/CSV Format/test_labels.csv')
        total_labels = pd.concat([train_labels, test_labels])
    
        remove_unnecessary_columns(total_labels, ['xmin', 'ymin', 'xmax', 'ymax'])

        total_labels['file_exists'] = [True if self._image_exists(lab) else False for lab in total_labels['filename']]
        total_labels['truth_label'] = [1 if 'military' in row else 0 for row in total_labels['class']]
        
        unsuitable_files = []
        for filename, metadata in total_labels.groupby('filename'):
            if _image_has_multiple_labels(metadata):
                unsuitable_files.append(filename)

        total_labels = total_labels[total_labels['file_exists'] & ~total_labels['filename'].isin(unsuitable_files)]
        return total_labels


def extract_labels_and_filenames(data):
    # Each image only has 1 unique label so just take the first
    return zip(*[(filename, row['truth_label'].iloc[0]) for filename, row in data.groupby('filename')])


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, config, batch_size=32, augment=True):
        'Initialization'
        self.batch_size = batch_size
        self.config = config
        self.filenames, self.ys = extract_labels_and_filenames(data)

        if augment:
            self.filenames = list(self.filenames)*config.augmentation_fac
            self.ys = list(self.ys) * config.augmentation_fac

        self.transform = make_transform(augment)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ys) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_ys        = self.ys[index*self.batch_size:(index+1)*self.batch_size]
        batch_filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        batch_images = [self.preprocess_image(name) for name in batch_filenames]

        return np.array(batch_images), np.array(to_categorical(batch_ys)).astype('float32')

    def preprocess_image(self, name):
        image = cv2.imread(f'{self.config.extract_path}/Images/{name}')
        image = Image.fromarray(image)
        processed_image = self.transform(image)
        processed_image = processed_image.cpu().detach().numpy()
        processed_image = np.moveaxis(processed_image, 0, -1)
        return cv2.resize(processed_image, (299, 299))


def create_data_generators(cfg, augment=True):
    dataloader = DataDownloader(cfg)

    train_generator = DataGenerator(dataloader.train, cfg, 256, augment=True if augment else False)
    val_generator   = DataGenerator(dataloader.val, cfg, 512, augment=True if augment else False)
    test_generator  = DataGenerator(dataloader.test, cfg, 512, augment=False)
    return train_generator, val_generator, test_generator
