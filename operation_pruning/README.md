# Operation Pruning

## Operation Pruning
To determine which operations will be pruned away: `python train_prune_operations.py`

## Retraining
- Store the path to the directory containing the model to be pruned in `retrain_pruned_network.py`
- Retrain the pruned network: `python retrain_pruned_network.py --arch="model_to_discretize"`

## Resume training
Set flag `model_to_resume` to resume search or evaluation. The settings will be automatically loaded from checkpoints.
- Resume search: `python train_prune_operations.py --model_to_resume="model_to_discretize"`
- Resume evaluation: `python retrain_pruned_network.py --model_to_resume="model_to_discretize"`