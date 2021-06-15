import numpy as np
import matplotlib.pyplot as plt

scaling_acc = [
    {
        "mu": 60,
        "acc": np.array([
            [97.309998, 97.399994, 97.369995], # search 1
            [97.430000, 97.639999, 97.349998], # search 2
            [97.320000, 97.269997, 97.509995]  # search 3
        ]),
        "model_size": np.array([
            [3.729718], # search 1
            [4.029598], # search 2
            [3.838150]  # search 3
        ]),
        "best_acc": np.array([
            [97.360001, 97.459999, 97.369995], # search 1
            [97.589996, 97.649994, 97.430000], # search 2
            [97.470001, 97.269997, 97.529999]  # search 3
        ])
    }
]

exp_index = 0

mu = scaling_acc[exp_index]["mu"]
print(u"GSparsity on search space S2:")

acc_final = scaling_acc[exp_index]["acc"].flatten()
model_size = scaling_acc[exp_index]["model_size"].flatten()
acc_best = scaling_acc[exp_index]["best_acc"].flatten()

"""without cutout in search"""
print(u"model size: {0:.2f} \u00B1 {1:.2f}".format(np.mean(model_size), np.std(model_size)))
print(u"Final acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_final), np.std(acc_final)))
print(u" Best acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_best), np.std(acc_best)))