import numpy as np


first_set = np.loadtxt("../datas/train_set.csv", str, delimiter=",", usecols=(6, 12), skiprows=-1)  # balance & duration
print(first_set)

