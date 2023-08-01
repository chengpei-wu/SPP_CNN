import os
import time

from sklearn.utils import shuffle
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model

from SpatialPyramidPooling import SpatialPyramidPooling
from utils import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # Ignore warning


class SPP_CNN:
    def __init__(self, epochs=30, batch_size=1, valid_proportion=0.1, lr=0.1, levels=[1, 2, 4],
                 model=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.metrics = ['mae', 'mse']
        self.lr = lr
        self.levels = levels
        self.optimizer = optimizers.SGD(learning_rate=self.lr)
        self.valid_proportion = valid_proportion
        if not model:
            # initial instance for training
            self.model = self.init_model()
        else:
            # initial instance for testing
            spp_layer = {'SpatialPyramidPooling': SpatialPyramidPooling}
            self.model = load_model(model, custom_objects=spp_layer)

    def init_model(self):
        model = Sequential()
        # note that the input_shape=(None, None, 1)
        # to allow variable input network sizes
        model.add(Conv2D(64, (7, 7), activation='relu',
                         input_shape=(None, None, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(SpatialPyramidPooling(self.levels))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(21, activation='hard_sigmoid'))
        model.compile(loss='mean_squared_error',
                      optimizer=self.optimizer,
                      metrics=self.metrics)
        model.summary()
        return model

    def data_generator(self, x, y):
        x, y = shuffle(x, y)
        batches = (len(x) + self.batch_size - 1) // self.batch_size
        while True:
            for i in range(batches):
                tt = x[i * self.batch_size:(i + 1) * self.batch_size]
                Y = y[i * self.batch_size:(i + 1) * self.batch_size]
                X = []
                for t in tt:
                    # adj matrix
                    adj = t.todense()
                    input_channels = generate_input_channels(
                        adj)
                    X.append(input_channels)
                X = np.array(X)
                yield X, Y

    def fit(self, x, y, model_path):
        filepath = f'{model_path}.hdf5'
        CheckPoint = ModelCheckpoint(filepath, monitor='val_mae', verbose=1, save_best_only=True, mode='min')
        # loss_history = Save_loss()
        on_Plateau = ReduceLROnPlateau(monitor='mse', patience=5, factor=0.5, min_delta=5e-4,
                                       verbose=1, min_lr=0.000001)
        callbacks_list = [CheckPoint, on_Plateau]
        x, y = shuffle(x, y)
        train_size = int(len(x) * (1 - self.valid_proportion))
        valid_size = len(x) - train_size
        x_train, y_train = x[:train_size], y[:train_size]
        x_valid, y_valid = x[train_size:], y[train_size:]
        self.model.fit_generator(
            generator=self.data_generator(x_train, y_train),
            steps_per_epoch=(train_size + self.batch_size -
                             1) // self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks_list,
            verbose=1,
            validation_data=self.data_generator(x_valid, y_valid),
            validation_steps=(valid_size + self.batch_size -
                              1) // self.batch_size
        )

    def my_predict(self, x):
        y_pred = []
        l = len(x)
        pred_times = []
        for i in range(l):
            print(
                '\r', f'predicting network robustness: {i} / {l}...', end='', flush=True)
            adj = x[i].todense()
            X = []
            # define channel
            input_channels = generate_input_channels(adj)
            X.append(input_channels)
            start_time = time.time()
            y_pred.append(self.model.predict(np.array(X)))
            end_time = time.time()
            pred_times.append(end_time - start_time)
        return np.array(y_pred).reshape(l, 21, 1)
