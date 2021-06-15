import numpy as np
import matplotlib.pyplot as plt

scaling_acc = [
    {
        "mu": 60,
        "acc": np.array([
            [97.339996, 97.399994, 97.180000], # search 1
            [97.059998, 97.259995, 96.979996], # search 2
            [96.919998, 97.029999, 97.009995]  # search 3
        ]),
        "model_size": np.array([
            [3.005254], # search 1
            [2.931382], # search 2
            [3.009790]  # search 3
        ]),
        "best_acc": np.array([
            [97.339996, 97.500000, 97.309998], # search 1
            [97.169998, 97.290001, 97.119995], # search 2
            [97.079994, 97.169998, 97.070000]  # search 3
        ])
    }
]

exp_index = 0

mu = scaling_acc[exp_index]["mu"]
print(u"PC DARTS:")

acc_final = scaling_acc[exp_index]["acc"].flatten()
model_size = scaling_acc[exp_index]["model_size"].flatten()
acc_best = scaling_acc[exp_index]["best_acc"].flatten()

"""without cutout in search"""
print(u"model size: {0:.2f} \u00B1 {1:.2f}".format(np.mean(model_size), np.std(model_size)))
print(u"Final acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_final), np.std(acc_final)))
print(u" Best acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_best), np.std(acc_best)))