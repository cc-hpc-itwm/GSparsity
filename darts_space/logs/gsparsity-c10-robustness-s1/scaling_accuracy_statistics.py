import numpy as np
import matplotlib.pyplot as plt

scaling_acc = [
    {
        "mu": 60,
        "acc": np.array([
            [97.149994, 96.939995, 96.889999], # search 1
            [96.709999, 96.869995, 96.799995], # search 2
            [97.159996, 97.019997, 96.930000]  # search 3
        ]),
        "model_size": np.array([
            [3.246058], # search 1
            [3.329650], # search 2
            [3.329650]  # search 3
        ]),
        "best_acc": np.array([
            [97.239998, 96.979996, 96.979996], # search 1
            [96.809998, 96.979996, 96.849998], # search 2
            [97.199997, 97.159996, 97.000000]  # search 3
        ])
    }
]

exp_index = 0

mu = scaling_acc[exp_index]["mu"]
print(u"GSparsity on search space S1:")

acc_final = scaling_acc[exp_index]["acc"].flatten()
model_size = scaling_acc[exp_index]["model_size"].flatten()
acc_best = scaling_acc[exp_index]["best_acc"].flatten()

"""without cutout in search"""
print(u"model size: {0:.2f} \u00B1 {1:.2f}".format(np.mean(model_size), np.std(model_size)))
print(u"Final acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_final), np.std(acc_final)))
print(u" Best acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_best), np.std(acc_best)))