# NAS on DARTS search space

## Search
Search for cell structures: `python train_search_group.py --weight_decay=xxx`
- The search for cell structures is performed in a small "search" network, and the cell structure will be scaled up to form a big network.
- Two types of cells are assumed: normal cell and reduce cell. Reduce cells are placed at 1/3 and 2/3 depth of the search network.
- `--weight_decay` specifies the regularization gain mu. The larger the regularization gain, the more operations will be pruned away (But there is a tradeoff between the accuracy and regularization gain).

## Evaluation
Evaluate the cell structure found in `Search`: `python scale_group_and_evaluate.py --arch="xxx"`
- Store the path to the directory containing the search network in `list_of_models.py` and assign a name such as `xxx`
- Prune the nonimportant operations in the above "search" network: Operations whose L2 norm are smaller than a threshold will be pruned from the search network. The threshold can be set by --pruning_threshold. 
- Note that the same operation in different cells of the same type will be pruned or preserved simultaneously, so that cells of the same type will have the same structure.
- After pruning, the normal cell and reduce cell in the "search" network will be scaled up to form a big network.
- The big network will be trained from scratch and the final accuracy will be used to judge the quality of the cell structure.

## Resume training
Set flag `path_to_resume` to resume search or evaluation. The settings will be automatically loaded from checkpoints.
- `--path_to_resume` specifies the path to the folder containing the model whose training should be resumed.
- Resume search: `python train_search_group --path_to_resume="xxx"`
- Resume evaluation: `python scale_group_and_evaluate.py --path_to_resume="xxx"`