└── Portfolio Optimization Library/
    ├── data_provider/
    │   ├── data_loader - Extract the data with dynamic inputs
    │   ├── data_factory - turn the data into the format of B x S x F
    │   ├── data_augment - select whether we need to augment the data/
    │   │   ├── frAug
    │   │   └── path_sig
    │   └── data_process - for feature engineering specialized for the model/
    │       ├── model 1
    │       └── model 2
    ├── model/
    │   ├── A
    │   └── B
    ├── exp/
    │   ├── forecast (a class including train, test, validation)
    │   └── basic that include the layout and all models
    ├── utils
    └── main.py