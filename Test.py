from SPP_CNN import SPP_CNN
from utils import *

# the trained model load path
model_load_path = f'./checkpoints/4(300,700)_nrnd_isd0_isw0_yc.hdf5'

# the testing data .mat file path
testing_data_path = f'./data/test/4(300,700)_nrnd_isd0_isw0.mat'

# the prediction output file saving path
prediction_saving_path = f'./prediction/4(300,700)_nrnd_isd0_isw0_yc.mat'

print(f'tesing infomation:\n\t'
      f'model: {model_load_path}\n\t'
      f'testing data: {testing_data_path}\n\t'
      f'saving path: {prediction_saving_path}')

x, y = load_network(testing_data_path, 'yc')
CNN = SPP_CNN(model=model_load_path)

y_pred = CNN.my_predict(x)
sio.savemat(prediction_saving_path, {'pred': y_pred, 'sim': y})
print(f'\nThe prediction results has saved to {prediction_saving_path}.')
