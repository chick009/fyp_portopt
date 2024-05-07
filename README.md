# Portfolio Allocation Baselines

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)


## Table of Contents

- [Background](#background)
- [Installation](#install)
- [Dataset](#dataset)
- [Folder Description](#usage)
- [Experiment](#run)


## Background 
Our Final Year Projects also include deep learning in portfolio optimization. Here, we applied the paper "Deep Learning for Portfolio Optimization" as a base, and expand it into two enhancements with signature methods. Our logic stays the same, preprocess the data, pass through hidden layers, and generate the future weights that optimize sharpe ratio. Our code structure generally follows the time series library.

## Installation
This project uses pip packages are enough, please be reminded the codes are run on half a year ago, 
so there may be compatibility issues in the installation, where our torch version is 2.0.1, with cuda 
version as 11.8, please check out https://pytorch.org/get-started/locally/ for more information.

```sh
$ pip install numpy pandas signatory yfinance joblib scipy matplotlib seaborn scikit-learn torch torchvision torchaudio torchviz umap-learn
```

## Dataset
We mainly pick 2 datasets for comparisons, including ETF Dataset suggested in paper, and fundamental MA chosen in our semester one
- ETF data: demo_close.csv
- Funadamental_MA: MA_stocks_v2.csv

## Folder Description:
- data_provider
    - augmentation.py: Augmentation using FrAug in batch
    - data_loader.py: Preprocess the data and turn it into dataloader for convenience (not the most efficient approach here)
- exp
    - portopt_DL_exp.py: Provide training, testing, prediction scheme for the model
- models
    - provides three sets of models that can be used for training/ testing/ prediction

## Experiments:

Run the shell code as of below, and we could input parameters in the main config.
```sh
$ python main.py
```