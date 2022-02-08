To test the performance of ResRep final models:

1. Download the final model `logs_resrep/latest.pth` of ResRep from Google Drive ([here](https://drive.google.com/drive/folders/1qvl_YcjVHd4Xus2Ck3VoccMwlf8D_bmJ?usp=sharing)), and save it to `logs_resrep/sres50_train_20220126-163758/`.

2. `python main_search_conv12_resrep.py

The reproduced top1 (top5) is 0.10% (0.56%), and the MACs of the pruned model is 1.86 GMac. The log of the reproduced experiment is summarized in `logs_resrep/sres50_train_20220126-163758/_log_reproduce.txt`.