# Operation Pruning
This folder contains the code and logs for pruning operations and retraining the pruned model.

The following result is obtained by performing operation pruning to `DARTS-V2` architecture (more details [here](https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/genotypes.py#L75)):
| regularization gain mu | accuracy before pruning | accuracy  after retraining | parameters pruned |
|:---:|:---:|:---:|:---:|
| 0 | 97.50 | N/A | 0% |
| 0.0001 | 96.50 | 97.45 | 21.75% |
| 0.0002 | 96.46 | 97.44 | 38.40% |
| 0.0005 | 96.36 | 97.32 | 47.91% |
| 0.002 | 96.47 | 97.09 | 60.46% |
| 0.004 | 96.48 | 96.84 | 67.20% |

## Operation Pruning
To determine which operations could be pruned away: `python train_prune_operations.py  --weight_decay=xxx`
- `--weight_decay` specifies the regularization gain mu. The larger the regularization gain, the more operations will be pruned away (But there is a tradeoff between the accuracy and regularization gain).

## Retraining
Retrain the pruned network: `python retrain_pruned_network.py --path_to_prune="xxx"`
- `--path_to_prune` specifies the path to the folder containing the model whose operations will be pruned.
- Operations whose L2 norm is smaller than a threshold will be removed. The threshold can be set by `--pruning_threshold`
- The pruned operations will be removed from the model. So the new model is smaller than the original model and faster to train.

## Resume training
Set flag `--path_to_resume` to resume the operation pruning or retraining. The settings will be automatically loaded from checkpoints.
- `--path_to_resume` specifies the path to the folder containing the model whose training should be resumed.
- Resume operation pruning: `python train_prune_operations.py --path_to_resume="xxx"`
- Resume retraining: `python retrain_pruned_network.py --path_to_resume="xxx"`