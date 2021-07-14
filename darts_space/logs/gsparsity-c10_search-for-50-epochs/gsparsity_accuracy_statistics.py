import numpy as np

acc_run1 = np.array([97.040001, 97.149994, 97.139999])
acc_run2 = np.array([97.279999, 97.199997, 97.229996])
acc_run3 = np.array([97.009995, 97.189995, 97.299995])

acc = np.concatenate((acc_run1, acc_run2, acc_run3))

print(u"GSparsity evaluation acc (cell found at epoch 50): {0:.2f} \u00B1 {1:.2f} (without cutout in search)".format(np.mean(acc), np.std(acc)))