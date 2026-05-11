
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from .features import get_audio_features
from .config import MAX_TIME_STEPS

class AudioDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size=32, augment=False, shuffle=True):
        super().__init__()
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_paths = [self.file_paths[k] for k in indexes]
        list_labels = [self.labels[k] for k in indexes]

        X, y = self.__data_generation(list_paths, list_labels)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_paths, list_labels):
        # Initialization
        # X1: MFCC, X2: MelSpectrogram (or vice versa depending on model input order)
        # We'll define Model Input 1: MelSpec (Rich texture)
        # Model Input 2: MFCC (Cepstral)
        
        # Dimensions based on config
        from .config import N_MFCC, N_MELS
        
        X_mfcc = np.empty((self.batch_size, N_MFCC, MAX_TIME_STEPS, 1))
        X_mel = np.empty((self.batch_size, N_MELS, MAX_TIME_STEPS, 1))
        y = np.empty((self.batch_size), dtype=int)

        for i, path in enumerate(list_paths):
            mfcc, mel = get_audio_features(path, augment=self.augment, target_length=MAX_TIME_STEPS)
            X_mfcc[i] = mfcc
            X_mel[i] = mel
            y[i] = list_labels[i]

        # Return dictionary mapping input layer names to data
        return {'input_mel': X_mel, 'input_mfcc': X_mfcc}, y
