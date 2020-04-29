import pandas as pd

bank_data = pd.read_csv("../datas/train_set.csv")
marital_set = list(bank_data['marital'])
education_set = list(bank_data['education'])
default_set = list(bank_data['default'])
housing_set = list(bank_data['housing'])
y_set = list(bank_data['y'])
data_set = []
for i in range(0, len(marital_set)):
    tmp = [marital_set[i], education_set[i], default_set[i], housing_set[i], y_set[i]]
    data_set.append(tmp)
label_set = bank_data['y']
label_set = list(label_set)
yes_count = 0
for i in label_set:
    if i == 1:
        yes_count += 1
no_count = len(marital_set)-yes_count
# 10-means cross validate
for i in range(0, 10):
    # divide the data set
    train_set = []
    test_set = []
    base1 = int(i*(yes_count/10))
    base2 = int((i+1)*(yes_count/10))
    base3 = yes_count + int(i*(no_count/10))
    base4 = yes_count + int((i+1)*(no_count/10))
    for j in range(0, base1):
        train_set.append(data_set[j])
    for j in range(base1, base2):
        test_set.append(data_set[j])
    if base2 < yes_count:
        for j in range(base2, yes_count):
            train_set.append(data_set[j])
    for j in range(yes_count, base3):
        train_set.append(data_set[j])
    for j in range(base3, base4):
        test_set.append(data_set[j])
    if base4 < yes_count+no_count:
        for j in range(base4, yes_count+no_count):
            train_set.append(data_set[j])
    # calculate begins
    train_len = len(train_set)
    test_len = len(test_set)
    print(test_len)
    print(test_set)
    print(train_set)
    tmp_no_count = 0
    tmp_yes_count = 0
    for j in train_set:
        if j[-1] == 0:
            tmp_no_count += 1
        else:
            tmp_yes_count += 1
    P_yes = tmp_yes_count/train_len
    P_no = tmp_no_count/train_len
    predict = 0
    for k in test_set:
        yes_mar = 0
        yes_edu = 0
        yes_def = 0
        yes_hsg = 0
        no_mar = 0
        no_edu = 0
        no_def = 0
        no_hsg = 0
        for t in train_set:
            if t[-1] == 0:  # no
                if t[0] == k[0]:
                    no_mar += 1
                if t[1] == k[1]:
                    no_edu += 1
                if t[2] == k[2]:
                    no_def += 1
                if t[3] == k[3]:
                    no_hsg += 1
            else:  # yes
                if t[0] == k[0]:
                    yes_mar += 1
                if t[1] == k[1]:
                    yes_edu += 1
                if t[2] == k[2]:
                    yes_def += 1
                if t[3] == k[3]:
                    yes_hsg += 1
        p_yes = yes_mar/tmp_yes_count*yes_edu/tmp_yes_count*yes_def/tmp_yes_count*yes_hsg/tmp_yes_count*P_yes
        # print(p_yes)
        p_no = no_mar/tmp_no_count*no_edu/tmp_no_count*no_def/tmp_no_count*no_hsg/tmp_no_count*P_no
        # print(p_no)
        if p_yes > p_no:
            if k[-1] == 1:
                predict += 1
        else:
            if k[-1] == 0:
                predict += 1
    # print(predict)
    score = predict/test_len
    print(score)
    with open("../output/scoresOfMyBayes.txt", "a") as sob:
        sob.write("The score of test "+str(i)+" is "+str(score)+'\n')
