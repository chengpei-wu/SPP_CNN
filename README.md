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