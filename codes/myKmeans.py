import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

frog_data = pd.read_csv("../datas/Frogs_MFCCs.csv")
mfcc_1 = frog_data['MFCCs_ 1']
mfcc_5 = frog_data['MFCCs_ 5']
mfcc_9 = frog_data['MFCCs_ 9']
mfcc_13 = frog_data['MFCCs_13']
mfcc_17 = frog_data['MFCCs_17']
mfcc_21 = frog_data['MFCCs_21']
first_set = []
for i in range(0, len(mfcc_1)):
    first_set.append([mfcc_1[i], mfcc_5[i], mfcc_9[i], mfcc_13[i], mfcc_17[i], mfcc_21[i]])
print(first_set)
