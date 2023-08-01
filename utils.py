import random

import networkx as nx
import numpy as np
import scipy.io as sio
import tensorflow.keras.callbacks
from PIL import Image
from scipy.sparse import csr_matrix


# save training loss
class Save_loss(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append([logs.get('mae'), logs.get('val_mae')])
        # if len(self.losses) > 1:
        #     print(f'improved: {self.losses[-1] - self.losses[-2]}')
        print(self.losses)


# load networks from .mat file
def load_network(path, robustness, fixed_size=None, sampling='bi'):
    """
    :param path: file path
    :param robustness: network robustness, 'yc' or 'lc', 'yc' means controllability robustness, 'lc' means connectivity robustness
    :param fixed_size: if using fixed size adjacency matrix
    :param sampling: the sampling method to get a fixed-size adjacency matrix
    :return: the network adjacency matrix and its robustness label
    """
    mat = sio.loadmat(path)
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    A = []
    y = []
    for i in range(len_net):
        for j in range(len_instance):
            print('\r',
                  f'loading {i * len_instance + j + 1} / {len_net * len_instance}  network...',
                  end='',
                  flush=True)
            if not fixed_size:
                A.append(adj_sort(networks[i, j]['adj'][0][0].todense()))
            else:
                if sampling == 'bi':
                    A.append(bilinear_interpolation_sampling(networks[i, j]['adj'][0][0].todense(), fixed_size))
                if sampling == 'rp':
                    A.append(ramdom_sampling(networks[i, j]['adj'][0][0].todense(), fixed_size))
            if robustness == 'lc':
                y.append(networks[i, j]['lc'][0][0])
            if robustness == 'yc':
                y.append(networks[i, j]['yc'][0][0])
    y = np.array(y)
    A = np.array(A)
    return A, y


# sort the adjacency matrix by node degrees
def adj_sort(adj):
    G = nx.from_numpy_matrix(adj)
    degrees = list(nx.degree(G))
    rank_degree = sorted(degrees, key=lambda x: x[1], reverse=True)
    rank_id = [i[0] for i in rank_degree]
    adj = np.array(adj)
    t_adj = adj[rank_id, :]
    t_adj = t_adj[:, rank_id]
    t_adj = csr_matrix(t_adj)
    return t_adj


# shuffle the adjacency matrix
def adj_shuffle(adj):
    s_adj = np.array(adj)
    terms = len(adj) // 2
    while terms:
        index1, index2 = random.randint(
            0, len(adj) - 1), random.randint(0, len(adj) - 1)
        # exchange rows
        s_adj[[index1, index2], :] = s_adj[[index2, index1], :]
        # exchange columns
        s_adj[:, [index1, index2]] = s_adj[:, [index2, index1]]
        terms -= 1
    return s_adj


# the bilinear interpolation algorithm, to unify and fix the adjacency matrix size
def bilinear_interpolation_sampling(adj, fixed_size):
    image = Image.new('L', (fixed_size, fixed_size))
    image.paste(Image.fromarray(np.uint8(adj)))
    resized_image = image.resize((fixed_size, fixed_size), Image.BILINEAR)
    result_matrix = np.array(resized_image, dtype=np.int)
    t_adj = csr_matrix(result_matrix)
    return t_adj


# the random sampling algorithm, to unify and fix the adjacency matrix size
def ramdom_sampling(adj, fixed_size):
    isd = 0
    size = len(adj)
    if isd:
        G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_matrix(adj)
    if size < fixed_size:
        n = fixed_size - size
        while n:
            G.add_node(size + n - 1)
            n -= 1
        A = nx.adjacency_matrix(G).todense()
        s_adj = csr_matrix(adj_shuffle(A))
    else:
        n = size - fixed_size
        rm_ids = np.random.choice(list(G.nodes()), size=(n), replace=False)
        G.remove_nodes_from(rm_ids)
        A = nx.adjacency_matrix(G).todense()
        s_adj = csr_matrix(adj_shuffle(A))
    return s_adj


# init input form for the CNN model
def generate_input_channels(adj):
    network_size = adj.shape[0]
    c = np.array(adj).reshape(network_size, network_size, 1)
    return c
