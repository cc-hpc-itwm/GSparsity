import numpy as np
import matplotlib.pyplot as plt

scaling_acc = [
    {
        "mu": 120,
        "acc": np.array([
            [84.139999, 83.669998, 83.869995], # search 1
            [83.329994, 83.549995, 83.709999], # search 2
            [83.610001, 83.110001, 83.019997]  # search 3
        ]),
        "model_size": np.array([
            [4.859920], # search 1
            [4.859920], # search 2
            [4.620016]  # search 3
        ]),
        "best_acc": np.array([
            [84.419998, 83.739998, 83.959999], # search 1
            [83.379997, 83.799995, 83.909996], # search 2
            [83.869995, 83.209999, 83.220001]  # search 3
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