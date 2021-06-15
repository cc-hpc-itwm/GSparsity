import numpy as np

acc_prestored = np.array([97.309998, 97.329994, 97.099998])

acc_run1 = np.array([96.970001, 96.899994, 97.059998]) # without cutout in search
acc_run2 = np.array([97.059998, 96.849998, 96.809998]) # without cutout in search
acc_run3 = np.array([96.930000, 96.970001, 97.019997]) #  without cutout in search

acc = np.concatenate((acc_run1, acc_run2, acc_run3))

print(u"DrNAS test acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc), np.std(acc)))
print(u"Prestored DrNAS test acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_prestored), np.std(acc_prestored)))