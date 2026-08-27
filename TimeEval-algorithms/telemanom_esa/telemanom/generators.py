import random
import numpy as np
random.seed(1992)
np.random.seed(1992)
from tensorflow.keras.utils import Sequence


class TelemanomGenerator(Sequence):
    def __init__(self, data: np.array, train_means, train_stds, input_channel_indices: list, target_channels_indices: list, window_size: int = 250,
                 prediction_window_size: int = 10, batch_size: int = 1, shuffle: bool = True,
                 prediction_mode: bool = False):
        self.data = data
        self.train_means = train_means
        self.train_stds = train_stds
        self.input_channel_indices = input_channel_indices
        self.target_channels_indices = target_channels_indices
        self.window_size = window_size
        self.prediction_window_size = prediction_window_size
        self.batch_size = batch_size
        self.prediction_mode = prediction_mode
        self.shuffle = shuffle

        # Compact index storage (int32 arrays) — a Python list of millions of
        # tuples uses hundreds of MB and contributed to execute OOMs.
        frag_ids = []
        offsets = []
        for i, arr in enumerate(self.data):
            # Remove fragments too short for training
            last_indices_to_remove = 0 if prediction_mode else self.prediction_window_size
            for j in range(len(arr) - self.window_size - last_indices_to_remove):
                frag_ids.append(i)
                offsets.append(j)
        self.frag_ids = np.asarray(frag_ids, dtype=np.int32)
        self.offsets = np.asarray(offsets, dtype=np.int32)
        self.nb_samples = int(self.frag_ids.shape[0])
        self._order = np.arange(self.nb_samples, dtype=np.int64)

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._order)

    def __len__(self):
        return int(np.ceil(self.nb_samples / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.nb_samples)
        order = self._order[start:end]

        # If needed - fill last batch with random indices during training. This is important when using BatchNorm
        if self.shuffle and start + self.batch_size > self.nb_samples:
            nb_to_add = self.batch_size - len(order)
            extra = np.random.choice(start, nb_to_add, replace=False)
            order = np.concatenate((order, self._order[extra]))

        n = len(order)
        n_in = len(self.input_channel_indices)
        means_in = self.train_means[self.input_channel_indices]
        stds_in = self.train_stds[self.input_channel_indices]
        input_data = np.empty((n, self.window_size, n_in), dtype=np.float32)

        if self.prediction_mode:
            for k, idx in enumerate(order):
                i = int(self.frag_ids[idx])
                j = int(self.offsets[idx])
                window_end = j + self.window_size
                input_data[k] = (self.data[i][j:window_end, self.input_channel_indices] - means_in) / stds_in
            return input_data

        n_out = len(self.target_channels_indices)
        means_out = self.train_means[self.target_channels_indices]
        stds_out = self.train_stds[self.target_channels_indices]
        output_data = np.empty((n, self.prediction_window_size, n_out), dtype=np.float32)
        for k, idx in enumerate(order):
            i = int(self.frag_ids[idx])
            j = int(self.offsets[idx])
            window_end = j + self.window_size
            input_data[k] = (self.data[i][j:window_end, self.input_channel_indices] - means_in) / stds_in
            output_data[k] = (self.data[i][window_end:window_end + self.prediction_window_size, self.target_channels_indices] - means_out) / stds_out
        return input_data, output_data
