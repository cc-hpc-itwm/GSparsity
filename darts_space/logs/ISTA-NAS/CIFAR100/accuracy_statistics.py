import numpy as np

acc_run1 = np.array([80.589996, 82.539997, 82.189997])
acc_run2 = np.array([83.259998, 83.319998, 82.939997])
acc_run3 = np.array([82.109997, 82.339997, 82.189997])


acc = np.concatenate((acc_run1, acc_run2, acc_run3)) 
acc_vertical = np.vstack((acc_run1, acc_run2, acc_run3))
best_arch_ind = np.argmax(np.mean(acc_vertical,axis=1))

print(u"Evaluation acc: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc), np.std(acc)))
print(u"Evaluation acc best model: {0:.2f} \u00B1 {1:.2f}".format(np.mean(acc_vertical,axis=1)[best_arch_ind], np.std(acc_vertical,axis=1)[best_arch_ind]))