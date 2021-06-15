import numpy as np
import matplotlib.pyplot as plt

scaling_acc = [
    {
        "mu": 60,
        "acc": np.array([
            [82.239998, 82.619995, 82.930000], # search 1: 38695
            [83.559998, 83.489998, 83.099998], # search 2: 79351
            [81.829994, 81.489998, 81.849998]  # search 3: 94775
        ]),
        "model_size": np.array([
            [3.646000], # search 1
            [4.089448], # search 2
            [2.737936]  # search 3
        ])
    }
]

exp_index = 0

mu = scaling_acc[exp_index]["mu"]
print(u"PC-DARTS on CIFAR-100:")

acc_final = scaling_acc[exp_index]["acc"].flatten()
model_size = scaling_acc[exp_index]["model_size"].flatten()

"""without cutout in search"""
print(u"model size: {0:.2f} \u00B1 {1:.2f}".format(np.mean(model_size), np.std(model_size)))
print(u"Final acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_final), np.std(acc_final)))