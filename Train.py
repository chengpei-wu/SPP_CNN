from SPP_CNN import SPP_CNN
from utils import *

# the training networks file path
training_data_path = f'./data/train/4(300,700)_nrnd_isd0_isw0.mat'

# the saving path of model
model_saving_path = f'./checkpoints/4(300,700)_nrnd_isd0_isw0_yc'

print(
    f'training infomation:\n\t'
    f'testing data: {training_data_path}\n\t'
    f'model saving path: {model_saving_path}\n'
)

x, y = load_network(training_data_path, 'yc')
CNN = SPP_CNN(
    epochs=20,
    batch_size=1,
    lr=1e-1,
    levels=[1, 2, 4],
    valid_proportion=0.1
)

CNN.fit(x=x, y=y, model_path=model_saving_path)
