# SPP-CNN: A Spatial Pyramid Pooling-based Convolutional Neural Network Approach for Network Robustness Prediction

## SPP-CNN
- SPP-CNN is proposed, which gains wider tolerance to different input sizes than the GNN-based approaches , while maintains fast approximation speed as the straightforward CNN-based approaches.
- SPP-CNN shows stronger generalization ability than the existing approaches on predicting the robustness of networks from unseen topologies and unseen sizes.
- SPP-CNN also shows stronger performance than the existing approaches on predicting the robustness of real-world networks. Additional validation experiment demonstrates the advantages of SPP-CNN is consistent when synthetic and real-world networks are put together for prediction

## Dependencies
- Python3, tensorflow(2.6.0)

- The code is based on tensorflow.keras, you need to install all the required packages.

- Other dependencies can be installed via `pip/conda install PackageName`

## Usage

- All parameters for running the program are pre-defined in `parameters.py`;

- All training or testing dataset are saved in `./data/train` or `./data/test`;

- The trained model is saved in `./models`;

- The predictions are saved in `./prediction`;

  You are free to change the dataset or model parameters according to your own tasks.

## Run demo

- To run the demo program of SPP-CNN,  you can run the `Train.py` and `Test.py` directly.

- Demo dataset explanation

  - The demo dataset is a mat file of 1000 networks form MULTI-REDDIT-12K.

  - The training label is the controllability robustness under random node-removal attacks. 

  - Network sizes are randomly distributed between 300-700.

  - All networks are undirected and unweighted.

## Cite

Please cite our paper if you use this code in your research work.

- [ ] Citation information of our paper will be updated soon.



# Usage

To make it easier for your dataset to be used directly in the code, it is recommended to keep the training data format
the same as in this code. Otherwise, you will need to modify the code in the data loading part to suit your own dataset.

## install

` conda install --yes --file requirements.txt`

## train

`python train.py`

## test

`python test.py`

## dataset

### 4(300,700)_nrnd_isd0_isw0.mat

- 4 types of networks: {ER, QS, SF, SW-NW}, each type include 500 instances, totally 2000 instances.
- For all networks, using random node removing attack (nrnd) to calculate the robustness.
- All network sizes range from 300 to 700.
- All networks are undirected(isd0) and unweighted(isw0).