# NAS on DARTS search space

## Search
Search: `python train_search_group.py`
- `train_search_group.py`: searches for a cell structure that will be scaled up to form the full network for evaluation (retraining).

## Evaluation
Evaluation: `python scale_group_and_evaluate.py --arch="model_to_discretize"`
- Store the path to the directory containing the model to be discretized in `list_of_models.py` and assign a name such as `model_to_discretize`
- `train_search_group`: scale the cell structure of `model_to_discretize` found in Search and train the network

## Resume training
Set flag `model_to_resume` to resume search or evaluation. The settings will be automatically loaded from checkpoints.
- Resume search: `python train_search_group --model_to_resume="model_to_discretize"`
- Resume evaluation: `python scale_group_and_evaluate.py --model_to_resume="model_to_discretize"`