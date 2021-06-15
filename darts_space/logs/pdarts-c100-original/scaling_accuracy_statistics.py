import numpy as np
import matplotlib.pyplot as plt

scaling_acc = [
    {
        "mu": 60,
        "acc": np.array([
            [83.330000, 83.200000, 83.120000], # search 1: 38695
            [83.620000, 83.380000, 83.850000], # search 2: 79351
            [83.330000, 83.550000, 83.760000]  # search 3: 94775
        ]),
        "model_size": np.array([
            [4.735936], # search 1
            [4.571776], # search 2
            [4.572856]  # search 3
        ])
    }
]

exp_index = 0

mu = scaling_acc[exp_index]["mu"]
print(u"P-DARTS on CIFAR-100:")

acc_final = scaling_acc[exp_index]["acc"].flatten()
model_size = scaling_acc[exp_index]["model_size"].flatten()

"""without cutout in search"""
print(u"model size: {0:.2f} \u00B1 {1:.2f}".format(np.mean(model_size), np.std(model_size)))
print(u"Final acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_final), np.std(acc_final)))