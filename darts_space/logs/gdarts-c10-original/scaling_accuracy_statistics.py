import numpy as np
import matplotlib.pyplot as plt

scaling_acc = [
    {
        "mu": 60,
        "acc": np.array([
            [96.57, 96.53, 96.41], # search 1: 38695
            [96.78, 96.84, 96.61], # search 2: 79351
            [96.32, 96.52, 96.45]  # search 3: 94775
        ]),
        "model_size": np.array([
            [2.85], # search 1
            [2.82], # search 2
            [2.55]  # search 3
        ])
    }
]

exp_index = 0

mu = scaling_acc[exp_index]["mu"]
print(u"G-DARTS:")

acc_final = scaling_acc[exp_index]["acc"].flatten()
model_size = scaling_acc[exp_index]["model_size"].flatten()

"""without cutout in search"""
print(u"model size: {0:.2f} \u00B1 {1:.2f}".format(np.mean(model_size), np.std(model_size)))
print(u"Final acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_final), np.std(acc_final)))