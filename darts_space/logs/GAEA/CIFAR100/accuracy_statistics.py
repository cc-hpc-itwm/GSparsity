import numpy as np

acc_run1 = np.array([78.500000, 78.070000, 78.099998])
acc_run2 = np.array([80.409996, 80.309998, 80.309998])
acc_run3 = np.array([79.269997, 79.400002, 79.329994])


acc = np.concatenate((acc_run1, acc_run2, acc_run3)) 
acc_vertical = np.vstack((acc_run1, acc_run2, acc_run3))
best_arch_ind = np.argmax(np.mean(acc_vertical,axis=1))

print(u"Evaluation acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc), np.std(acc)))
print(u"Evaluation acc best model: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_vertical,axis=1)[best_arch_ind], np.std(acc_vertical,axis=1)[best_arch_ind]))