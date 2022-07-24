import numpy as np
import scipy.io as sio
import tensorflow.keras.callbacks
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from parameters import isd, isw


class Save_loss(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        if len(self.losses) > 1:
            print(f'improved: {self.losses[-1] - self.losses[-2]}')
        print(self.losses)


def load_network(path, roubustness):
    mat = sio.loadmat(path)
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    A = []
    y = []
    for i in range(len_net):
        for j in range(len_instance):
            print('\r',
                  f'loading { i * len_instance + j + 1} / {len_net * len_instance}  network...',
                  end='',
                  flush=True)
            A.append(networks[i, j]['adj'][0][0])
            if roubustness == 'lc':
                y.append(networks[i, j]['lc'][0][0])
            if roubustness == 'yc':
                y.append(networks[i, j]['yc'][0][0])
    y = np.array(y)
    A = np.array(A)
    return A, y


def generate_input_channels(adj):
    network_size = adj.shape[0]
    c = np.array(adj).reshape(network_size, network_size, 1)
    return c
