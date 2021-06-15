import numpy as np
import matplotlib.pyplot as plt

scaling_acc = [
    {
        "mu": 60,
        "acc": np.array([
            [97.309998, 97.209999, 97.309998], # search 1
            [97.269997, 97.259995, 97.589996], # search 2
            [97.489998, 97.329994, 97.500000]  # search 3
        ]),
        "model_size": np.array([
            [4.029598], # search 1
            [3.729718], # search 2
            [4.029598]  # search 3
        ]),
        "best_acc": np.array([
            [97.419998, 97.379997, 97.409996], # search 1
            [97.309998, 97.339996, 97.639999], # search 2
            [97.570000, 97.500000, 97.519997]  # search 3
        ])
    }
]

exp_index = 0

mu = scaling_acc[exp_index]["mu"]
print(u"GSparsity on search space S4:")

acc_final = scaling_acc[exp_index]["acc"].flatten()
model_size = scaling_acc[exp_index]["model_size"].flatten()
acc_best = scaling_acc[exp_index]["best_acc"].flatten()

"""without cutout in search"""
print(u"model size: {0:.2f} \u00B1 {1:.2f}".format(np.mean(model_size), np.std(model_size)))
print(u"Final acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_final), np.std(acc_final)))
print(u" Best acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_best), np.std(acc_best)))