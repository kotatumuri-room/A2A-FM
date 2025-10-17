# How to prepare datasets
First prepare the pretrained autoencoders before doing this process.
## ZINC22
1. access [ZINC22](https://cartblanche.docking.org/tranches/2d) and download files under `./ZINC22`
1. run ```python script/datasets/preprocess/zinc_preprocess_logp_tpsa.py```
1. run ```python script/datasets/preprocess/zinc_subsample_split_low_high.py```

## CelebADialogHQ
1. Download the dataset from [github](https://github.com/ziqihuangg/CelebA-Dialog) under `./celebADialogHQ`
1. run ```python script/datasets/preprocess/embed_celba_hq.py```
1. split the data by running ```python script/datasets/preprocess/celeba_val_split```