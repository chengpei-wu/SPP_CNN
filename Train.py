from CNN import CNN
from utils import *
from parameters import *

# the tarining data .mat file path
training_data_path = f'./data/train/{training_network_scale}_{attack_strategy}_isd{isd}_isw{isw}.mat'
# the saving path of model
model_saving_path = f'./models/{training_network_scale}' \
    f'_{attack_strategy}_isd{isd}_isw{isw}_{training_robustness}'


print(f'training infomation:\n\t'
      f'testing data: {training_data_path}\n\t'
      f'model saving path: {model_saving_path}')

x, y = load_network(training_data_path, training_robustness)
spp_cnn = CNN(epochs=epochs, batch_size=batch_size,
          valid_proportion=valid_proportion)
spp_cnn.fit(x=x, y=y, model_path=model_saving_path)
