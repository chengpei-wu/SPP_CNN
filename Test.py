from CNN import CNN
from utils import *
from parameters import *

# the model loading path
model_load_path = f'./models/{training_network_scale}' \
                  f'_{attack_strategy}_isd{isd}_isw{isw}_{training_robustness}'

# the testing data .mat file path
testing_data_path = f'./data/test/{testing_network_scale}' \
                    f'_{attack_strategy}_isd{isd}_isw{isw}.mat'

# the prediction output file saving path
prediction_saving_path = f'./prediction/{training_network_scale}' \
                         f'_{attack_strategy}_isd{isd}_isw{isw}_{training_robustness}_test{testing_network_scale}.mat'

print(f'tesing infomation:\n\t'
      f'model: {model_load_path}\n\t'
      f'testing data: {testing_data_path}\n\t'
      f'saving path: {prediction_saving_path}')

x, y = load_network(testing_data_path, training_robustness)
spp_cnn = CNN(model=f'{model_load_path}.hdf5')

y_pred = spp_cnn.my_predict(x)
sio.savemat(prediction_saving_path, {'pred': y_pred, 'sim': y})
print(f'The prediction results has saved to {prediction_saving_path}.')
