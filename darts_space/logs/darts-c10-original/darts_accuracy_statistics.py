import numpy as np

acc_run1 = np.array([96.739998, 96.909996, 96.909996])
acc_run2 = np.array([97.059998, 97.180000, 96.959999])
# acc_run3 = np.array([97.299995, 97.299995, 97.349998])
acc_run4 = np.array([96.889999, 96.809998, 96.939995])

acc = np.concatenate((acc_run1, acc_run2, acc_run4)) # RUN3 is abandoned because its run is interrupted.

print(u"DARTS evaluation acc (cell found at epoch 100): {0:.2f} \u00B1 {1:.2f} (without cutout in search)".format(np.mean(acc), np.std(acc)))

acc_run1_epoch50 = np.array([96.820000, 97.139999, 96.799995])
acc_run2_epoch50 = np.array([97.199997, 97.079994, 96.979996])
acc_run4_epoch50 = np.array([96.930000, 97.029999, 96.879997])

acc_epoch50 = np.concatenate((acc_run1_epoch50, acc_run2_epoch50, acc_run4_epoch50))

print(u"DARTS evaluation acc (cell found at epoch 050): {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_epoch50), np.std(acc_epoch50)))
