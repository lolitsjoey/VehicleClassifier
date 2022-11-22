import os
import shutil
from itertools import chain
import random
from urllib.request import urlopen
from zipfile import ZipFile
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import torch
from sklearn.model_selection import train_test_split
from PIL import Image
import albumentations as aug
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

from src.augmentation import make_transform, ScaleIntensities
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
    
        remove_unnecessary_columns(total_labels, [])

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


def cv2_augment():
    transform = aug.Compose([
        aug.HorizontalFlip(p=0.25),
        aug.VerticalFlip(p=0.25),
        aug.GaussNoise(p=0.15),
        aug.GaussianBlur(p=0.15),
        aug.RandomBrightnessContrast(p=0.2),
        aug.RandomShadow(p=0.2),
        aug.RandomRain(p=0.2)
    ], p=1)
    return transform


def get_batch_idxs(filenames, hf_bz, index):
    a = (index * hf_bz) % len(filenames)
    b = ((index + 1) * hf_bz) % len(filenames)

    if a > b:
        return chain(range(a, len(filenames)), range(b))
    return range(a, b)


def plot_batch_images(batch_images, batch_ys, batch_filenames):
    for image, y, filename in zip(batch_images, batch_ys, batch_filenames):
        flat = image.ravel()
        fig, axes = plt.subplots(2)
        axes[0].imshow(image)
        axes[1].imshow(ScaleIntensities([np.min(image[:, :, 0]), np.max(image[:, :, 0])], [0, 255])(image[:, :, 0]).astype(np.int32))
        plt.suptitle(f'{y}  -  {filename}')
        plt.show()


def shuffle(batch_images, batch_filenames, batch_ys):
    zipped = list(zip(batch_images, batch_filenames, batch_ys))
    random.shuffle(zipped)
    return [list(i) for i in zip(*zipped)]


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, config, augment=True):
        'Initialization'
        self.batch_size = config.batch_size
        self.config = config
        self.data = data
        self.filenames, self.ys = extract_labels_and_filenames(data)

        if augment:
            self.filenames = list(self.filenames)*config.augmentation_fac
            self.ys = list(self.ys) * config.augmentation_fac

        self.filenames_class0 = [filename for idx, filename in enumerate(self.filenames) if self.ys[idx] == 0]
        self.filenames_class1 = [filename for idx, filename in enumerate(self.filenames) if self.ys[idx] == 1]

        self.transform = make_transform(augment, config)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ys) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        hf_bz = int(self.batch_size // 2)
        batch_ys        = [0] * hf_bz + [1] * hf_bz
        batch_filenames = [self.filenames_class0[i] for i in get_batch_idxs(self.filenames_class0, hf_bz, index)]\
                        + [self.filenames_class1[i] for i in get_batch_idxs(self.filenames_class1, hf_bz, index)]
        batch_images = [self.preprocess_image(name) for name in batch_filenames]

        batch_images, batch_filenames, batch_ys = shuffle(batch_images, batch_filenames, batch_ys)
        plot_batch_images(batch_images, batch_ys, batch_filenames)
        return np.array(batch_images), np.array(to_categorical(batch_ys)).astype('float32')

    def preprocess_image(self, name):
        image = cv2.imread(f'{self.config.extract_path}/Images/{name}')
        aug_object = cv2_augment()
        transformed_image = aug_object(image=image)['image']

        bbox = self.data.loc[self.data.filename == name, ['xmin', 'xmax', 'ymin', 'ymax']].values[0]

        processed_image = self.pytorch_transforms(transformed_image, bbox)
        return cv2.resize(processed_image, (self.config.im_dim, self.config.im_dim))

    def pytorch_transforms(self, transformed_image, bbox):
        image = Image.fromarray(transformed_image)

        processed_image = self.transform(image)

        processed_image = processed_image.cpu().detach().numpy()
        processed_image = np.moveaxis(processed_image, 0, -1)
        return processed_image


def create_data_generators(cfg, augment=True):
    dataloader = DataDownloader(cfg)

    train_generator = DataGenerator(dataloader.train, cfg, augment=True if augment else False)
    val_generator   = DataGenerator(dataloader.val, cfg, augment=True if augment else False)
    test_generator  = DataGenerator(dataloader.test, cfg, augment=False)
    return train_generator, val_generator, test_generator
