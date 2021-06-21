# NAS on NB201 search space

## Search
- `train_search_group`: searches for a cell structure on the NasBench-201 model. Three datasets can be run with this script:
    - CIFAR-10. This is the default dataset.
    - CIFAR-100. This dataset can be run by setting the flag `--dataset='cifar100'`.
    - ImageNet16-120. This dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4). Run it using the flag `--dataset='imagenet16-120'`.

- run `train_search_group.py`: `python train_search_group.py --dataset='cifar100'`