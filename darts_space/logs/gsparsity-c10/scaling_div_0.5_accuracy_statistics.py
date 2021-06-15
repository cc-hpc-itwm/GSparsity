import numpy as np
import matplotlib.pyplot as plt

scaling_acc = [
    {
        "mu": 60,
        "acc": np.array([
            [97.269997, 97.000000, 97.169998], # search 1
            [96.949997, 97.159996, 97.169998], # search 2
            [97.269997, 97.290001, 97.229996]  # search 3
        ]),
        "model_size": np.array([
            [4.735414], # search 1
            [4.316662], # search 2
            [4.556566]  # search 3
        ]),
        "best_acc": np.array([
            [97.349998, 97.129997, 97.299995], # search 1
            [97.019997, 97.169998, 97.229996], # search 2
            [97.339996, 97.360001, 97.299995]  # search 3
        ])
    }
]

exp_index = 0

mu = scaling_acc[exp_index]["mu"]
print(u"Normalization by divide with exponential 0.5 and mu {:2d}:".format(mu))

acc_final = scaling_acc[exp_index]["acc"].flatten()
model_size = scaling_acc[exp_index]["model_size"].flatten()
acc_best = scaling_acc[exp_index]["best_acc"].flatten()

"""without cutout in search"""
print(u"model size: {0:.2f} \u00B1 {1:.2f}".format(np.mean(model_size), np.std(model_size)))
print(u"Final acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_final), np.std(acc_final)))
print(u" Best acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_best), np.std(acc_best)))