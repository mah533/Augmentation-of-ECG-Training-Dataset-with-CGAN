"""
this is a classifier
input is the normalized ECG vector
output is the class of the ECG

adapted from https://www.youtube.com/watch?v=Jy4wM2X21u0
YT, by Aladdin Personn, 2020

"""
import copy
import csv
import json
import os
import gc
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime, timedelta
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from ekg_class import dicts
import torch.nn.functional as F
import torch.nn as nn
from models_classifier import EcgResNet34
from sklearn.metrics import classification_report as report
from sklearn.metrics import confusion_matrix as cf_matrix
from sklearn.metrics import roc_auc_score

from utils import print_confusion_matrix

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

num2descr, letter2num, letter2descr, num2letter = dicts()
start_time = datetime.now()

print(("\n" + "*" * 61 + "\n\t\t\t\t\tstart time  {0:02d}:{1:02d}:{2:02.0f}\n" + "*" * 61).format(
    start_time.hour, start_time.minute, start_time.second))

drive = "F:"
# drive = "E:"
myPath_base = os.path.join(drive, "\\UTSA\\ECG_Synthesis\\dell_g7\\PycharmProjects\\paper2_data\\MIT_BIH")
# myPath_base = os.path.join(drive, "\\UTSA\\ECG_Synthesis\\dell_g7\\PycharmProjects\\paper3_data")

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dry_run = False
BATCH_SIZE = 16
CHANNELS_IMG = 1
# INPUT_SIZE = 256
# INPUT_SIZE = 140
LEARNING_RATE = 1e-3

if dry_run:
    NUM_EPOCHS = 1
else:
    NUM_EPOCHS = 10

aug_key = "augmented"
# aug_key = ""

# cond_key = "uncond"
cond_key = "cond"

# screen_key = "screened"
screen_key = "not_screened"


tr_ts_ratio = 0.9  # train/test split ratio
len_ratio = 1  # to study the effect of support set number of samples (shorter train sets)
if '.' in str(len_ratio):
    len_ratio_str = str(len_ratio).replace('.', '')
else:
    len_ratio_str = str(len_ratio)
# for reducing number of samples in training set
# len_ratio used in section:        %%% begin build X, y (training and test sets)  %%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
stats_all_classes = 
        {
        'N': 75052,   'L': 8075,  'R': 7259,  'V': 7130,
        '/': 7028,    'A': 2546,  'f': 982,   'F': 803,
        'j': 229,     'a': 150,   'E': 106,   'J': 83,
        'Q': 33,      'e': 16,    'S': 2
        }

'/' and 'P' are the same        
'''

# all the 15 classes
all_classes = ['/', 'A', 'E', 'F', 'J', 'L', 'N', 'Q', 'R', 'S', 'V', 'a', 'e', 'f', 'j']
all_classes.remove('Q')
all_classes.remove('S')
# all_classes.remove('V')
# why remove 'V'? plots show no-uniform pattern like other classes, resemble cl Q

# %%%%%%%%%%%%%%%%  begin select classes %%%%%%%%%%%%%%%
# classes2keep = copy.copy(all_classes)
classes2keep = ['/', 'A', 'L', 'N', 'R', 'f', 'j']

classes2keep_folder = copy.copy(classes2keep)
idx = classes2keep_folder.index('/')
classes2keep_folder.pop(idx)
classes2keep_folder.insert(idx, 'P')

NUM_CLASSES = len(classes2keep)
# %%%%%%%%%%%%%%%%  end select classes %%%%%%%%%%%%%%%
# myPath_save = os.path.join(myPath_base, "trained_classifiers\\00_{}\\{}_len_ratio_{}_{}_{}".
#                           format(cond_key, classes2keep_folder, len_ratio_str, aug_key, screen_key))

path = "gb_{}\\{}".format(cond_key, screen_key)
myPath_save = os.path.join(myPath_base, path)
os.makedirs(myPath_save, exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  begin ECG500 dataset %%%%%%%%%%%%%%%%
"""
myPath_read_ECG5000 = os.path.join(myPath_base, "Datasets\\UCRArchive_2018_time_series\\ECG5000")

with open(os.path.join(myPath_read_ECG5000, "ECG5000_TEST.tsv"), 'r') as f:
    temp = csv.reader(f, delimiter='\t')
    train_set_temp = list(temp)

train_set = []
for idx in range(len(train_set_temp)):
    train_set.append(list(map(float, train_set_temp[idx])))

with open(myPath_base+"ECG5000_TRAIN.tsv", 'r') as f:
    temp = csv.reader(f, delimiter='\t')
    test_set_temp = list(temp)

test_set = []
for idx in range(len(test_set_temp)):
    test_set.append(list(map(float, test_set_temp[idx])))


y_train = [int(train_set[idx][0])-1 for idx in range(len(train_set))]
X_train = []
for idx in range(len(train_set)):
    X_train.append(train_set[idx][1:])
y_train_stats = Counter(y_train)

y_test = [int(test_set[idx][0])-1 for idx in range(len(test_set))]
X_test = []
for idx in range(len(test_set)):
    X_test.append(test_set[idx][1:])

y_test_stats = Counter(y_test)
NUM_CLASSES = len(y_train_stats)
a = 0
"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  end ECG500 dataset  %%%%%%%%%%%%%%%%%

keys = [classes2keep_folder, str(len_ratio_str), aug_key, cond_key, screen_key]
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  begin MIT-BIH dataset   %%%%%%%%%%%%%%%%%%

myPath_dataset = os.path.join(drive, '\\UTSA\\ECG_Synthesis\\dell_g7', "Datasets\\mitbih_datasets_Dictionaries")
with open(os.path.join(myPath_dataset, "record_X_y_adapt_win_bef075_aft075_Normalized.json"), "r") as f:
    data = json.load(f)
stats_all_classes = Counter(np.asarray(data, dtype=object)[:, 2])

vals = [[] for idx in range(NUM_CLASSES)]

data_rl_dict = dict(zip(classes2keep, vals))

# create a dictionary of data which are in classes2keep
for item in data:
    if item[2] in classes2keep:
        data_rl_dict[item[2]].append(item[1])
del data

if '/' in classes2keep:
    temp = data_rl_dict['/']
    data_rl_dict.pop('/')
    data_rl_dict['P'] = temp

# %%%%%%%%%%%%%%%%%%%%%  begin build data2keep_dict from genbeats  (cond.) %%%%%%%%%%%%%%%%%%%%%
"""
# Cond. GAN
# for reading genbeats
path_gb = "PycharmProjects\\paper2_gen_data\\MIT_BIH\\multiclass\\genbeats_ac_wgan_gp_{}_len_ratio_{}". \
    format(classes2keep_folder, len_ratio_str)

myPath_read_gb = os.path.join(myPath_base, path_gb)

data_gb_dict_keys = copy.deepcopy(classes2keep_folder)
data_gb_dict_keys.pop(data_gb_dict_keys.index('N'))
data_gb_dict_vals = []

for cl in data_gb_dict_keys:
    f_name = "gb_cond_{}.json".format(cl)
    with open(os.path.join(myPath_read_gb, f_name), 'r') as f:
        temp = json.load(f)
    data_gb_dict_vals.append(temp[:5000])

data_gb_dict = dict(zip(data_gb_dict_keys, data_gb_dict_vals))

for cl in ['A', 'f', 'j']:
    f_name = "gb_cond_{}_2.json".format(cl)
    with open(os.path.join(myPath_read_gb, f_name), 'r') as f:
        temp = json.load(f)
    data_gb_dict[cl].extend(temp[:5000])

# augment "data_gb_dict" to "data_rl_dict"
data2keep_dict = copy.deepcopy(data_rl_dict)

temp_key = copy.deepcopy(classes2keep_folder)
temp_key.pop(temp_key.index('N'))

# augment real data with genbeats
for key in temp_key:
    data2keep_dict[key].extend(data_gb_dict[key])

# trim the lengths to 10000
for key in data2keep_dict.keys():
    temp = data2keep_dict[key][0:10000]
    data2keep_dict[key] = temp

a = 0
del data_rl_dict, data_gb_dict, data_gb_dict_keys, data_gb_dict_vals, item, temp, temp_key, vals
"""
# %%%%%%%%%%%%%%%%%%%%%  end build data2keep_dict from genbeats  (cond.) %%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%  begin build data2keep_dict from genbeats  (Uncond.) %%%%%%%%%%%%%%%%%%%%%
# Unconditional GAN
# for reading genbeats


# build data_gb_dict, no genbeats for class 'N'
data_gb_dict_keys = copy.deepcopy(classes2keep_folder)
data_gb_dict_keys.pop(data_gb_dict_keys.index('N'))
data_gb_dict_vals = [[] for _ in range(len(classes2keep_folder))]
data_gb_dict = dict(zip(data_gb_dict_keys, data_gb_dict_vals))

for cl in data_gb_dict_keys:
    myPath_read_gb = os.path.join(myPath_base, "gb_{}\\{}".format(cond_key, screen_key))
    f_name = "gb_{}_{}_{}.json".format(cl, cond_key, screen_key)
    with open(os.path.join(myPath_read_gb, f_name), 'r') as f:
        temp = json.load(f)
    data_gb_dict[cl].extend(temp)

# augment "data_gb_dict" to "data_rl_dict"
data2keep_dict = copy.deepcopy(data_rl_dict)


# augment real data with genbeats
for cl in classes2keep_folder:
    if cl == 'N':
        continue
    val_length = len(data2keep_dict[cl])
    data2keep_dict[cl].extend(data_gb_dict[cl][:10000-val_length])

# trim class 'N' to the lengths of 10000
data2keep_dict['N'] = data2keep_dict['N'][:10000]

del data_rl_dict, data_gb_dict, data_gb_dict_keys, data_gb_dict_vals, item, temp, vals
# %%%%%%%%%%%%%%%%%%%%%  end build data2keep_dict from genbeats (Uncond.) %%%%%%%%%%%%%%%%%%%%%%%


vals = [[] for idx in range(NUM_CLASSES)]  # needed because they are filled in data_rl_dict for loop
data_train_dict = dict(zip(classes2keep_folder, vals))

vals = [[] for idx in range(NUM_CLASSES)]
data_test_dict = dict(zip(classes2keep_folder, vals))

# randomly splitting the dataset into train and test sets
for key in data2keep_dict.keys():
    val_len = len(data2keep_dict[key])
    idx_train = torch.randperm(val_len)[:int(tr_ts_ratio * val_len)]
    idx_test = torch.randperm(val_len)[int(tr_ts_ratio * val_len) + 1:]
    data_train_dict[key] = [data2keep_dict[key][idx] for idx in idx_train]
    data_test_dict[key] = [data2keep_dict[key][idx] for idx in idx_test]

# %%%%%%%%%%%%%%%%%     begin save sample plots of classes in classes2keep    %%%%%%%%%%%%%%%%%%%%
'''
for cl in classes2keep:
    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.suptitle("Class {} ({}: {}), count: {}".
                 format(classes2keep.index(cl), cl, letter2descr[cl], len(data2keep_dict[cl])))

    count = 0
    for i in range(3):
        for j in range(3):
            count += 1
            if count >= len(data2keep_dict[cl]):
                continue
            axes[i][j].plot(data2keep_dict[cl][count])
            axes[i][j].grid()
    plt.savefig(os.path.join(myPath_save, "00_sample_cl_{}.png".format(classes2keep.index(cl))))

plt.close("all")
'''
# %%%%%%%%%%%%%%%%%     end save sample plots of classes in classes2keep      %%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%    begin build X, y (training and test sets)    %%%%%%%%%%%%%%%%%%%%
X_train = []
y_train = []
# data_train_dict contains all the training data
# X_train contains only a portion of training data (len_ratio), to study of size of training set on 
#           classification metrics

a = 0
for key in data_train_dict.keys():
    temp = data_train_dict[key]
    length = int(len_ratio * len(temp))
    X_train.extend(temp[:length])
    idx = [classes2keep_folder.index(key)] * length
    # print('key {}, idx {}'.format(key, idx))
    y_train.extend(idx)

X_test = []
y_test = []
for key in data_test_dict.keys():
    X_test.extend(data_test_dict[key][:])
    idx = [classes2keep_folder.index(key)] * len(data_test_dict[key])
    y_test.extend(idx)

y_test_stat = Counter(y_test)
y_train_stat = Counter(y_train)
# %%%%%%%%%%%%%%%%%     end build X, y (training and test sets)    %%%%%%%%%%%%%%%%%%%%

keys = [classes2keep_folder, str(len_ratio_str), aug_key, cond_key, screen_key]
# %%%%%%%%%%%%%%%%%     begin save X, y (training and test sets)    %%%%%%%%%%%%%%%%%%%%
'''
with open(os.path.join(myPath_save, "X_train.json"), "w") as f:
    json.dump(X_train, f)
with open(os.path.join(myPath_save, "y_train.json"), "w") as f:
    json.dump(y_train, f)
with open(os.path.join(myPath_save, "X_test.json"), "w") as f:
    json.dump(X_test, f)
with open(os.path.join(myPath_save, "y_test.json"), "w") as f:
    json.dump(y_test, f)

del data2keep_dict, data_test_dict, data_train_dict
'''
# %%%%%%%%%%%%%%%%%     end save X, y (training and test sets)    %%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   end of MIT-BIH dataset  %%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% begin load Train & Test datasets %%%%%%%%%%%%%%%%%%%%%%%
"""
with open(os.path.join(myPath_save, "X_train_{}_len_ratio_{}.json".format(*keys)), "r") as f:
    X_train = json.load(f)
with open(os.path.join(myPath_save, "y_train_{}_len_ratio_{}.json".format(*keys)), "r") as f:
    y_train = json.load(f)
with open(os.path.join(myPath_save, "X_test_{}_len_ratio_{}.json".format(*keys)), "r") as f:
    X_test = json.load(f)
with open(os.path.join(myPath_save, "y_test_{}_len_ratio_{}.json".format(*keys)), "r") as f:
    y_test = json.load(f)
"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end load Train & Test datasets %%%%%%%%%%%%%%%%%%%%%%%
a = 0

dataset_train = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

dataset_test = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

"""
# MNIST: load data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

"""

# %%%%%%%%%%%%%%%%%%%%      Select and Initialize Network       %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# net = net_cnn(num_classes=NUM_CLASSES).to(device)
# net = net_fc(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
net = EcgResNet34(num_classes=NUM_CLASSES).to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# %%%%%%%%%%%%%%%%%%%%    begin  Train Network       %%%%%%%%%%%%%%%%%%%%%%%%%%
'''
for epoch in range(NUM_EPOCHS):
    # running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.reshape(inputs.shape[0], 1, -1)

        # print("labels = {}".format(labels))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        scores = net(inputs)
        loss = criterion(scores.squeeze(), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        if batch_idx % 200 == 0:  # print every 200 mini-batches
            now = datetime.now()
            print('{:02d}:{:02d}:{:02d}\t\tepoch={:4d} / {:4d}\t\titer={:5d} / {:5d}\t\t\tloss: {:7.5f}'.
                  format(now.hour, now.minute, now.second, epoch, NUM_EPOCHS, batch_idx, len(train_loader), loss))

print('\n\tFinished Training\n\n\n')
'''
# %%%%%%%%%%%%%%%%%%%%    end  Train Network       %%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%    begin save model   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
f_name = "00_classifier_{}_{}.pth".format(cond_key, screen_key)

PATH = os.path.join(myPath_save, f_name)
torch.save(net.state_dict(), PATH)

# display some samples from test set
# dataiter = iter(test_loader)
# images, labels = dataiter.next()
'''
# %%%%%%%%%%%%%%%%%%%%    end save model     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%    load trained classifier     %%%%%%%%%%%%%%%%%%%%%

classifier_name = "00_classifier_real_imbal_{}_len_ratio_{}.pth".format(classes2keep_folder, len_ratio_str)
net.load_state_dict(torch.load(os.path.join(myPath_save, classifier_name)))

del X_test, y_test, X_train, y_train
del dataset_train, dataset_test
torch.cuda.empty_cache()
gc.collect()
# torch.cuda.memory_summary(device=None, abbreviated=False)


# %%%%%%%%%%%%%%%%%%    print Train Set Report to file      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
original_stdout = sys.stdout
with open(os.path.join(myPath_save, 'res_train_{}_{}.txt'.format(cond_key, screen_key)), 'w') as f:
    sys.stdout = f

    scores = []
    y_true = []
    y_pred = []
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.reshape(inputs.shape[0], 1, -1).to(device)
        labels = labels.to(device)

        # output = net(inputs).max(dim=1)

        temp = net(inputs)
        output = temp.max(dim=1)

        y_true.extend(labels.data.tolist())
        y_pred.extend(output.indices.tolist())

    print("\n")
    print("%" * 20 + "\tClassification Report (Training Set, {})\t".format(aug_key) + "%" * 20)
    print("Classes used: \t\t{}".format(classes2keep_folder))
    print("Classifier Model:\t{}".format(net._get_name()))
    print(f"number of epochs:\t{NUM_EPOCHS}\n")
    print("Train/Test ratio {}:\t".format(tr_ts_ratio))
    print("Train set reduction length ratio: {}\t".format(len_ratio_str))
    print("Augmentation: {}".format(aug_key))
    print("Screening with DTW: {}".format(screen_key))
    print("Conditional: {}\n\n".format(cond_key))
    print(report(y_true, y_pred, target_names=classes2keep_folder))
    print("\nConfusion Matrix:\n {}".format(cf_matrix(y_true, y_pred)))
    print_confusion_matrix(cf_matrix(y_true, y_pred), class_names=classes2keep_folder,
                           fig_name="Conf. Matrix (Training Set)")
    plt.savefig(os.path.join(myPath_save, "cfmx_train_{}_{}.png".format(*keys)))


# %%%%%%%%%%%%%%%%%%    print Test Set Report to file  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

with open(os.path.join(myPath_save, 'res_test_{}_{}.txt'.format(cond_key, screen_key)), 'w') as f:
    sys.stdout = f

    scores = []
    y_true = []
    y_pred = []
    for batch_idx, (inputs, labels) in enumerate(test_loader, 0):
        inputs = inputs.reshape(inputs.shape[0], 1, -1).to(device)
        labels = labels.to(device)

        output = net(inputs).max(dim=1)

        y_true.extend(labels.data.tolist())
        y_pred.extend(output.indices.tolist())

    print("\n")
    print("%" * 20 + "\tClassification Report (Test Set, {})\t".format(aug_key) + "%" * 20)
    print("Classes used: \t\t{}".format(classes2keep_folder))
    print("Classifier Model:\t{}".format(net._get_name()))
    print(f"number of epochs:\t{NUM_EPOCHS}\n")
    print("Train/Test ratio: {}\t".format(tr_ts_ratio))
    print("Train set reduction length ratio: {}\t".format(len_ratio_str))
    print("Augmentation: {}".format(aug_key))
    print("Screening with DTW: {}".format(screen_key))
    print("\nConditional: {}\n\n".format(cond_key))
    print(report(y_true, y_pred, target_names=classes2keep_folder))
    print("Confusion Matrix:\n {}".format(cf_matrix(y_true, y_pred)))
    print_confusion_matrix(cf_matrix(y_true, y_pred), class_names=classes2keep_folder,
                           fig_name="Conf. Matrix (Test Set")
    plt.savefig(os.path.join(myPath_save, "cfmx_test_{}_{}.png".format(cond_key, screen_key)))


sys.stdout = original_stdout

finish_time = datetime.now()
print(("\n\n\n" + "finish time = {0:02d}:{1:02d}:{2:02.0f}").format(
    finish_time.hour, finish_time.minute, finish_time.second))

laps = finish_time - start_time
tot_sec = laps.total_seconds()
h = int(tot_sec // 3600)
m = int((tot_sec % 3600) // 60)
s = int(tot_sec - (h * 3600 + m * 60))

print("total elapsed time = {:02d}:{:2d}:{:2d}".format(h, m, s))

"""
#load model
model = model_name(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""
