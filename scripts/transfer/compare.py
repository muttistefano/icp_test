from platform import python_branch
import numpy as np
import matplotlib.pyplot as plt

loss_train_pre  = np.load("loss_train_pre_xyw.npy")
loss_valid_pre  = np.load("loss_valid_pre_xyw.npy")

loss_train_fine = np.load("loss_train_fine_xyw.npy")
loss_valid_fine = np.load("loss_valid_fine_xyw.npy")

loss_test_pre = np.load("test_eval_pre_fine.npy")


plt.figure("pre")
plt.plot(loss_train_pre,label="train")
plt.plot(loss_valid_pre,label="valid")
plt.yscale("log")
plt.legend()

plt.figure("fine")
plt.plot(loss_train_fine,label="train")
plt.plot(loss_valid_fine,label="valid")
plt.legend()
plt.yscale("log")


plt.figure("test")
plt.plot(loss_test_pre.T)

plt.show()
