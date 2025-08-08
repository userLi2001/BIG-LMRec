
## 1. Data
In argument '--data', 'afk' refers to Amazon Food-Kitchen dataset, 'amb' refers to Amazon Movie-Book dataset, and 'abe' refers to Amazon Beauty-Electronics dataset.

The code for the original data is available at: [https://cseweb.ucsd.edu/\~jmcauley/datasets/amazon/links.html]


## 2. Usage
Please check demo.sh on running on different datasets.


## 3. File Tree

    BIG_LMRec/
    ├── data/
    │   ├── abe/
    │   │   ├── abe_50_preprocessed.txt
    │   │   ├── abe_50_seq.pkl
    │   │   ├── map_item.txt
    │   │   └── map_user.txt
    │   ├── afk/
    │   │   ├── afk_50_preprocessed.txt
    │   │   ├── afk_50_seq.pkl
    │   │   ├── map_item.txt
    │   │   └── map_user.txt
    │   └── amb/
    │       ├── amb_50_preprocessed.txt
    │       ├── map_item.txt
    │       └── map_user.txt
    ├── dataloader.py
    ├── demo.sh
    ├── main.py
    ├── models/
    │   ├── BIG_LMRec.py
    │   ├── encoders.py
    │   ├── fuse_layer.py
    │   ├── Capsule.py
    │   └── layers.py
    ├── README.md
    ├── trainer.py
    └── utils/
        ├── constants.py
        ├── metrics.py
        ├── misc.py
        ├── noter.py
        └── preprocess.py


## 4. environments
python >=3.8
torch >=2.1.0
