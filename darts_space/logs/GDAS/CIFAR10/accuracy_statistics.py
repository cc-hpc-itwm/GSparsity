import numpy as np

acc_run1 = np.array([96.73, 96.56, 96.55])
acc_run2 = np.array([96.89, 96.60, 96.81])
acc_run3 = np.array([96.70, 96.69, 96.93])


acc = np.concatenate((acc_run1, acc_run2, acc_run3)) 
acc_vertical = np.vstack((acc_run1, acc_run2, acc_run3))
best_arch_ind = np.argmax(np.mean(acc_vertical,axis=1))

print(u"Evaluation acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc), np.std(acc)))
print(u"Evaluation acc best model: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_vertical,axis=1)[best_arch_ind], np.std(acc_vertical,axis=1)[best_arch_ind]))