To test the performance of ResRep final models:

1. Download the final model `logs_resrep/latest.pth` of ResRep from Google Drive ([here](https://drive.google.com/drive/folders/1qvl_YcjVHd4Xus2Ck3VoccMwlf8D_bmJ?usp=sharing)), and save it to `logs_resrep/sres50_train/`.

2. `python main_search_conv12_resrep.py --resume="logs_resrep"`