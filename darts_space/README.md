Search: `python train_search_group`
- `train_search_group`: searches for a cell structure that will be scaled up to form the full network for evaluation (retraining).

Evaluation: `python scale_group_and_evaluate.py --arch='xxx''
- `train_search_group`: scale the cell structure found in Search and train the network.

Set flag `model_to_resume` to resume search or evaluation. The settings will be automatically loaded from checkpoints.

- Resume search: `python train_search_group --model_to_resume=xxx`

- Resume evaluation: `python scale_group_and_evaluate.py --model_to_resume='xxx''