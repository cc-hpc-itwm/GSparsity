- `train_search_group`: searches for a group structure that will be scaled up to form the full network for evaluation (retraining). Depending on the singularity of the group, a group may consist of a cell, a stage or an operation:
    - search for a cell (by putting the same operation in different cells in the same group): normal cell and reduction cell. The found normal cell and reduction cell can be scaled up to form the full network.
    - search for a stage (by putting same operation in different stages in the same group). An example of the stage is: normal_cell normal_cell (stage_normal 1) reduce_cell (stage_redyce 1) normal_cell normal_cell(stage_normal 2) reduce_cell (stage_reduce 2) normal_cell normal_cell (stage_normal 3).
    - search for an operation, which is equivalent to pruning operations. As a result, the operations remained in each cell could be different.

- `train_prune_operations`: prune operations in full DARTS network (20 cells)

- run `scale_group_and_evaluate.py`: `python scale_group_and_evaluate.py --arch='cell_half_mu_50' --model_to_resume='testtt/testtt-lr_0.025_momentum_0.9_wd_0.0003_cells_14_time_20210301-195713'`