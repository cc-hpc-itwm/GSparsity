import numpy as np

acc_run1 = np.array([63.879997, 79.290001, 80.229996])
acc_run2 = np.array([80.790001, 81.029999, 81.339996])
acc_run3 = np.array([65.400002, 65.070000, 65.199997])

acc = np.concatenate((acc_run1, acc_run2, acc_run3))
acc_vertical = np.vstack((acc_run1, acc_run2, acc_run3))
best_arch_ind = np.argmax(np.mean(acc_vertical,axis=1))

print(u"DARTS evaluation acc (cell found at epoch 50): {0:.2f} \u00B1 {1:.2f} (without cutout in search)".format(np.mean(acc), np.std(acc)))
print(u"Evaluation acc best model: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_vertical,axis=1)[best_arch_ind], np.std(acc_vertical,axis=1)[best_arch_ind]))