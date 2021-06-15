# Filter pruning
This branch is created to train model with ProxSGD optimizer for Filter Pruning.

## Setup
Kindly follow the following steps to get experimental results.

1. Create pytorch enviornment.
2. Pass arguments as mentioned in main_filter_pruning.py.
3. Change variable lr, momentum, weight_decay values for different Experiments and Run main_filter_pruning.py 
4. Results can be seen in directory mentioned in arguments.

## Search for optimal value of mu
Kindly follow the following steps to get experimental results using optuna.

1. Create pytorch enviornment and install Optuna library.
2. Pass arguments as mentioned in search_mu_filter_pruning.py.
3. Change variable n_trials for number of experiment and update range weight_decay values for different Experiments and Run search_mu_filter_pruning.py 
4. Results can be seen in directory mentioned in arguments.



