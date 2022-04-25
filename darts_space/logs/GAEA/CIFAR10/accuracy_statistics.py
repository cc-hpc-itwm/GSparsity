import numpy as np

acc_run1 = np.array([95.949997, 95.849998, 96.189995])
acc_run2 = np.array([96.149994, 96.079994, 96.110001])
acc_run3 = np.array([96.519997, 96.629997, 96.570000])


acc = np.concatenate((acc_run1, acc_run2, acc_run3)) 
acc_vertical = np.vstack((acc_run1, acc_run2, acc_run3))
best_arch_ind = np.argmax(np.mean(acc_vertical,axis=1))

print(u"Evaluation acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc), np.std(acc)))
print(u"Evaluation acc best model: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_vertical,axis=1)[best_arch_ind], np.std(acc_vertical,axis=1)[best_arch_ind]))