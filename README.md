# SPP-CNN: A Spatial Pyramid Pooling-based Convolutional Neural Network Approach for Network Robustness Prediction

## Background 
We design a CNN-based complex network robustness predictor that allows to train networks with different sizes together, both contrallability and connectivity robustness are supported. 

Compared to the previous approaches(CNN-RP, PATCHY-SAN, LFR-CNN), it guarantees both very short prediction time and arbitrary sizes of network input without compromising accuracy.

In addition, it also obtains the state-of-arts generalisation ability.

## Install
the code is based on tf.keras, you need to import all the required packages.

`pip/conda install packagename `

## Usage
All parameters for running the program are pre-set in parameters.py;
All training or testing dataset are saved in ./data/train or ./data/test;
The trained model is saved in ./models;
The predictions are saved in ./prediction;

you are free to change the dataset or model parameters according to your own tasks.