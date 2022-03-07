# Filter pruning on ImageNet and ResNet-50

This implementation performs filter pruning on ResNet-50 and ImageNet.

- This implementation is based on the official PyTorch implementation of ImageNet and ResNet-50 ([here](https://github.com/pytorch/examples/tree/master/imagenet)).

- This implementation is tailored to ResNet-50. Modifications are needed for other architectures.

## Preparation
- For distributed training configurations, please refer to the [original implementation](https://github.com/pytorch/examples/tree/master/imagenet).
- `args.dist_url` may need to be modified based on the computing environment.
- The conda environment used to run this experiment is summarized in `environment.yml` (but probably not all libraries are needed to run this particular experiment).

## Searching

To determine which filters could be pruned, run

- if only filters of conv1 and conv2 in each block are prunable **[x]**:\
`python main_search_conv12.py --weight_decay=0.05 --normalization="div" --pretrained`


- if all filters of conv1-3 in each block are prunable:\
`python main_search_all_conv.py --weight_decay=0.09 --normalization="div" --pretrained`

- Note that **[x]** means that this is the approach adopted in the paper.

- To resume searching, run\
`python main_search_conv12.py --resume='path_to_experiment'` **[x]**, or\
`python main_search_all_conv.py --resume='path_to_experiment'`

An example of `path_to_experiment` is `conv12_lr_0.001_momentum_0.9_wd_0.05_normalization_div_pretrained_True_20211102-092334`.

- After searching with ProxSGD is completed, the filters whose L2 norm are smaller than `pruning_threshold` are pruned and the full model is automatically compressed to a small dense model. The weights of the nonzero filters are also passed to the compressed model.

- Save the path to the compressed model in `list_of_models.py`, and assign a name, such as `pretrained_conv12_layer_adaptive_mu_0_05`.

## Retraining
Retrain the compressed model, whose initial weights are inherited from the weights found by ProxSGD in the above search phase.

Retrain a model, run

- if only filters of conv1 and conv2 in each block are pruned **[x]**:\
`python main_reduce_retrain_conv12.py --baseline_model="pretrained_conv12_layer_adaptive_mu_0_05"`

- if all filters of conv1-3 in each block are pruned:\
`python main_reduce_retrain_all_conv.py --baseline_model="xxx"`

- To resume retraining, run\
`python main_reduce_retrain_conv12.py --resume='path_to_experiment'` **[x]**, or\
`python main_reduce_retrain_all_conv.py --resume='path_to_experiment'`

An example of `path_to_experiment` is `retrain_resnet50_pretrained_conv12_layer_adaptive_mu_0_05_20211105-074034`.

## Remarks

- To our best knowledge, this is the **first** implementation in which the zero filters are pruned (rather than masked) from the original model.

- How the models are compressed (illustrated by the example of pruning filters of conv1 and conv2 in each block):
1. The mask is computed based on which filters are zero and which filters are nonzero, using the function `compute_and_save_mask` in `main_search_conv12.py`.
2. Infer from mask the number of nonzero channels and build the compressed model in `models_with_reduce_conv12.py`
3. Infer from mask the indices of nonzero filters, and transfer the parameters of the nonzero filters from the full model to the compressed model using the function `transfer_model_parameters` in `main_search_conv12.py`.

- To determine the acc of the original ResNet-50 model, run\
`python main_original.py --pretrained --evaluate`

- The original log files of GSparsity are reorganized into `logs_gsparsity_search` or `logs_gsparsity_retrain` for the sake of a clean repository structure.

- To directly use the checkpoint of GSparsity, please move the folder containing the searching or retraining experiment to the same level as the python scripts (i.e., one level above).

- Due to the file size limit of GitHub and space limit of Google Drive, only the checkpoints of mu={0.02, 0.05, 0.07, 0.1} are uploaded to Google Drive ([here](https://drive.google.com/drive/folders/1qvl_YcjVHd4Xus2Ck3VoccMwlf8D_bmJ?usp=sharing)). Other checkpoints will be uploaded when a practical solution is found, or per request.

- `ptflops` is used to calculate the model complexity (see [here](https://github.com/sovrasov/flops-counter.pytorch) for more information).
