import numpy as np
import matplotlib.pyplot as plt

scaling_acc = [
    {
        "mu": 60,
        "acc": np.array([
            [83.419998, 83.129997, 83.549995], # search 1: 38695
            [83.430000, 83.619995, 83.180000], # search 2: 79351
            [83.25000, 83.029999, 83.189995]  # search 3: 94775
        ]),
        "model_size": np.array([
            [4.593160], # search 1
            [4.664656], # search 2
            [4.601152]  # search 3
        ])
    }
]

exp_index = 0

mu = scaling_acc[exp_index]["mu"]
print(u"DrNAS on CIFAR-100:")

acc_final = scaling_acc[exp_index]["acc"].flatten()
model_size = scaling_acc[exp_index]["model_size"].flatten()

"""without cutout in search"""
print(u"model size: {0:.2f} \u00B1 {1:.2f}".format(np.mean(model_size), np.std(model_size)))
print(u"Final acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_final), np.std(acc_final)))