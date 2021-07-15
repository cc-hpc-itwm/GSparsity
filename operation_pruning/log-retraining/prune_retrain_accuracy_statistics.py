import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 12.5}

matplotlib.rc('font', **font)

result_reduce = [
    {
        "mu": 0, #baseline, full network DARTS_v2
        "model_size": 3.349394, # in MB
        "acc": [97.529999, 97.489998, 97.470001]
    },
    {
        "mu": 0.0001,
        "model_size": 2.620898,
        "acc_reduce": [97.409996, 97.500000, 97.439995]
    },
    {
        "mu": 0.0002,
        "model_size": 2.063328,
        "acc_reduce": [97.459999, 97.549995, 97.299995]
    },
#     {
#         "mu": 0.0003,
#         "model_size": 2.05152,
#         "acc_reduce": [97.459999, 97.369995, 97.299995]
#     },
#     {
#         "mu": 0.0004,
#         "model_size": 2.003496,
#         "acc_reduce": [97.379997, 97.399994, 97.369995]
#     },
    {
        "mu": 0.0005,
        "model_size": 1.744798,
        "acc_reduce": [97.309998, 97.369995, 97.269997],
        "acc_freeze": [97.290001]
    },
#     {
#         "mu": 0.001,
#         "model_size": 1.489918,
#         "acc_reduce": [97.229996, 97.259995, 97.000000],
#         "acc_freeze": [97.110001]
#     },
    {
        "mu": 0.002,
        "model_size": 1.324318,
        "acc_reduce": [97.059998, 97.070000, 97.129997]
    },
#     {
#         "mu": 0.003,
#         "model_size": 1.255306,
#         "acc_reduce": [97.009995, 97.089996, 97.089996]
#     },
    {
        "mu": 0.004,
        "model_size": 1.09874,
        "acc_reduce": [96.829994, 96.889999, 96.790001]
    },
#     {
#         "mu": 0.005,
#         "model_size": 1.064792,
#         "acc_reduce": [97.019997, 96.989998, 96.779999],
#         "acc_freeze": [96.930000]
#     },
#     {
#         "mu": 0.006,
#         "model_size": 1.0622,
#         "acc_reduce": [96.899994, 96.589996, 96.970001]
#     },
#     {
#         "mu": 0.007,
#         "model_size": 1.014284,
#         "acc_reduce": [96.720001, 97.189995, 96.799995],
#         "acc_freeze": [96.579994]
#     },
#     {
#         "mu": 0.008,
#         "model_size": 1.0127,
#         "acc_reduce": [96.589996, 96.540001, 96.970001],
#         "acc_freeze": [96.720001]
#     },
#     {
#         "mu": 0.010,
#         "model_size": 0.969644,
#         "acc_reduce": [96.399994, 96.540001, 96.729996],
#         "acc_freeze": [96.589996]
#     },
    {
        "mu": 0.009,
        "model_size": 0.948938,
        "acc_reduce": [96.889999, 96.489998, 96.509995],
        "acc_freeze": [96.500000]
    }
]

model_size_baseline = result_reduce[0]['model_size']
acc_baseline = np.mean(result_reduce[0]['acc'])

sparsity_list = [0]
acc_list = [acc_baseline]
num_data_points = len(result_reduce) - 1
for k in range(1, num_data_points):
    data = result_reduce[k]
    sparsity_list.append((1-data['model_size']/model_size_baseline)*100)
    acc_list.append(np.mean(data['acc_reduce']))
    print("mu: {}, parameters pruned: {}, acc: {}".format(data['mu'], sparsity_list[-1], acc_list[-1]))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlim([0, 80])
ax.set_ylim([90, 100])

plt.xlabel("Percentage of weights pruned (in %)")
plt.ylabel("Accuracy")
plt.grid(True)

plt.plot(sparsity_list, acc_list, marker="8")
for i, j in zip(sparsity_list, acc_list):
    ax.annotate('%.2f' % j, xy=(i,j))

plt.savefig("operation_pruning_sparsity_vs_acc.eps", format="eps")
