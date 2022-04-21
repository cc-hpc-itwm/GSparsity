import numpy as np

acc_run1 = np.array([81.47, 80.93, 81.23])
acc_run2 = np.array([80.80, 81.44, 80.90])
acc_run3 = np.array([81.91, 81.39, 80.93])


acc = np.concatenate((acc_run1, acc_run2, acc_run3)) 
acc_vertical = np.vstack((acc_run1, acc_run2, acc_run3))
best_arch_ind = np.argmax(np.mean(acc_vertical,axis=1))

print(u"Evaluation acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc), np.std(acc)))
print(u"Evaluation acc best model: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_vertical,axis=1)[best_arch_ind], np.std(acc_vertical,axis=1)[best_arch_ind]))