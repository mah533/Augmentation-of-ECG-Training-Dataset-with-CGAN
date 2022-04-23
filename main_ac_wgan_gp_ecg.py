"""

"""
import copy
import csv
import datetime
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from ekg_class import dicts
from model_ac_wgan_gp_ecg import Disc_ac_wgan_gp_1d, Gen_ac_wgan_gp_1d, initialize_weights_1d
from utils import gradient_penalty, normalize, grid_plot_save, grid_plot

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

num2descr, letter2num, letter2descr, num2letter = dicts()
start_time = datetime.datetime.now()
print(("\n" + "*" * 50 + "\n\t\tstart time:      {0:02d}:{1:02d}:{2:02.0f}\n" + "*" * 50).format(
    start_time.hour, start_time.minute, start_time.second))

drive = "F:"
drive = "E:"
myPath_base = os.path.join(drive, "\\UTSA\\ECG_Synthesis\\dell_g7")
myPath_dataset = os.path.join(myPath_base, "Datasets\\mitbih_datasets_Dictionaries\\")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dry_run = False
if dry_run:
    NUM_EPOCHS = 1
else:
    NUM_EPOCHS = 30
BATCH_SIZE = 16
CHANNELS_IMG = 1
CRITIC_ITERATIONS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
GEN_EMBEDDING = 100
LAMBDA_GP = 10
LEARNING_RATE = 1e-4
Z_DIM = 100

tr_ts_ratio = 0.9     # "training set / test set" split ratio
len_ratio = 0.5     # to study the effect of support set number of samples (shorter train sets)
if '.' in str(len_ratio):
    len_ratio_str = str(len_ratio).replace('.', "")
else:
    len_ratio_str = str(len_ratio)

# transforms = transforms.Compose(
#    [
#        transforms.Resize(IMG_SIZE),
#        transforms.ToTensor(),
#        transforms.Normalize(
#            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
#    ]
# )

with open(os.path.join(myPath_dataset, "record_X_y_adapt_win_bef075_aft075_Normalized.json"), "r") as f:
    data = json.load(f)

stats_all_classes = Counter(np.asarray(data, dtype=object)[:, 2])

all_classes = ['/', 'A', 'E', 'F', 'J', 'L', 'N', 'Q', 'R', 'S', 'V', 'a', 'e', 'f', 'j']
all_classes_folder = ['P', 'A', 'E', 'F', 'J', 'L', 'N', 'Q', 'R', 'S', 'V', 'a', 'e', 'f', 'j']

all_classes.remove('Q')
all_classes.remove('S')
# all_classes.remove('V')      # why remove this? plots show no-uniform pattern

# %%%%%%%%%%%%%%%%%    begin classes statistics      %%%%%%%%%%%%%%%%%%
"""
stats_all_classes = 
        {
        'N': 75052,   'L': 8075,  'R': 7259,  'V': 7130,
        '/': 7028,    'A': 2546,  'f': 982,   'F': 803,
        'j': 229,     'a': 150,   'E': 106,   'J': 83,
        'Q': 33,      'e': 16,    'S': 2
        }
P: Paced beat
A: Atrial Premature contraction
E: Ventricular Escape beat
F: Fusion of Ventricular and Normal beat
J: Nodal (junctional) Premature Beat
L: Left bundle branch block beat
N: Normal beat
Q: Unclassifiable beat
R: Right bundle branch block beat
S: Premature or ectopic supraventricular beat
V: Premature Ventricular Contraction
a: Aberrated Atrial Premature beat
e: Atrial escape beat
f: Fusion of paced and normal beat
j: Nodal (junctional) escape beat
"""
# %%%%%%%%%%%%%%%%%    end classes statistics    %%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%       begin MIT-BIH Dataset      %%%%%%%%%%%%%%%%%%%%%%%%%%%%

d_set = "MIT_BIH"
# Note: '/' and 'P' are the same
classes2keep = ['/', 'A', 'L', 'N', 'R', 'f', 'j']
classes2keep_folder = ['P', 'A', 'L', 'N', 'R', 'f', 'j']

# d_set = "ECG5000\\"
# classes2keep = [1, 2, 3, 4, 5]

myPath_save = os.path.join(myPath_base, "PycharmProjects\\paper2_gen_data\\", d_set, "multiclass",
                            "genbeats_ac_wgan_gp_cl_{}_len_ratio_{}".format(classes2keep_folder, len_ratio_str))
os.makedirs(myPath_save, exist_ok=True)

NUM_CLASSES = len(classes2keep)
IMG_SIZE = 256

with open(os.path.join(myPath_dataset, "record_X_y_adapt_win_bef075_aft075_Normalized.json"), "r") as f:
    data = json.load(f)

stats_all_classes = Counter(np.asarray(data, dtype=object)[:, 2])

# create dictionary of data to be kept
vals = []
for idx in range(len(classes2keep)):
    vals.append([])

data2keep_dict = dict(zip(classes2keep, vals))
data_train_dict = copy.deepcopy(data2keep_dict)
data_test_dict = copy.deepcopy(data2keep_dict)

for item in data:
    if item[2] in classes2keep:
        data2keep_dict[item[2]].append(item[1])
del data

if '/' in classes2keep:
    temp = data2keep_dict['/']
    data2keep_dict.pop('/')
    data2keep_dict['P'] = temp

# randomly splitting the dataset into train and test sets
for key in classes2keep_folder:
    val_len = len(data2keep_dict[key])
    idx_train = torch.randperm(val_len)[:int(tr_ts_ratio * val_len)]
    idx_test = torch.randperm(val_len)[int(tr_ts_ratio * val_len)+1:]
    data_train_dict[key] = [data2keep_dict[key][idx] for idx in idx_train]
    data_test_dict[key] = [data2keep_dict[key][idx] for idx in idx_test]


# plot samples of beats in classes2keep
for cl in classes2keep_folder:
    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.suptitle("Class {} ({}: {}), count:{}".format(classes2keep_folder.index(cl), cl, letter2descr[cl],
                                                      len(data2keep_dict[cl])))

    count = 0
    for i in range(3):
        for j in range(3):
            count += 1
            if count >= len(data2keep_dict[cl]):
                continue
            axes[i][j].plot(data2keep_dict[cl][count])
            axes[i][j].grid()
    # plt.savefig(os.path.join(myPath_save, "00_sample_cl_{}".format(classes2keep_folder.index(cl))))

plt.close("all")

a = 0
# %%%%%%%%%%%%%%%%%%%%%%%%% begin save templates %%%%%%%%%%%%%%%%%%%%%%%
"""
# recall: classes2keep = ['N', 'L', 'R', 'V', 'P', 'A']
cl = "V"
path_class = os.path.join(my_path_base, "MIT_BIH")
template_folder = "templates"
f_name = "template_{}_revB.json".format(cl)
num_templates = 50

templates = data_train_dict[cl][:num_templates]
os.makedirs(os.path.join(path_class, template_folder), exist_ok=True)
with open((os.path.join(path_class, template_folder, f_name)), "w") as f:
    json.dump(templates, f)
a = 0
"""
# %%%%%%%%%%%%%%%%%%%%%%%%% end save templates %%%%%%%%%%%%%%%%%%%%%%%%%


'''
X = []
y = []
for item in data:
    if item[2] in classes2keep:
        X.append(item[1])
        idx = classes2keep.index(item[2])
        y.append(idx)
stats_y = Counter(y)
'''

# %%%%%%%%%%%%%%%%%    begin build X, y (training and test sets)    %%%%%%%%%%%%%%%%%%%%
X_train = []
y_train = []
# data_train_dict contains all the training data
# X_train contains only a portion of training data (len_ratio), to study of size of training set on
#           classification metrics

a = 0
for key in classes2keep_folder:
    temp = data_train_dict[key]
    length = int(len_ratio*len(temp))
    X_train.extend(temp[:length])
    idx = [classes2keep_folder.index(key)] * length
    # print('key {}, idx {}'.format(key, idx))
    y_train.extend(idx)

X_test = []
y_test = []
for key in classes2keep_folder:
    X_test.extend(data_test_dict[key][:])
    idx = [classes2keep_folder.index(key)] * len(data_test_dict[key])
    y_test.extend(idx)

y_test_stat = Counter(y_test)
y_train_stat = Counter(y_train)

print("train: {}".format(y_train_stat))
print("test: {}".format(y_test_stat))
a = 0

# %%%%%%%%%%%%%%%%%     end build X, y (training and test sets)    %%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%       end MIT-BIH Dataset      %%%%%%%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% begin ECG500 data set %%%%%%%%%%%%%%%%%%%%%%%
"""
# {N: 58381, V: 69, S: 148, Q: 13, r: 1107}
# d_set = "ECG5000\\"             # classes in ECG5000: N, r, S, Q , V
# classes2keep = [1, 2, 3, 4, 5]

IMG_SIZE = 140
myPath_data = "E:\\UTSA\\ECG_Synthesis\\dell_g7\\Datasets\\UCRArchive_2018_time_series\\ECG5000\\"

# read train set, create train_set list
with open(myPath_data+ "ECG5000_TEST.tsv", 'r') as f:
    temp = csv.reader(f, delimiter='\t')
    train_set_temp = list(temp)
train_set = []
for idx in range(len(train_set_temp)):
    train_set.append(list(map(float, train_set_temp[idx])))

# creat test set, create test_set list
with open(myPath_data + "ECG5000_TRAIN.tsv", 'r') as f:
    temp = csv.reader(f, delimiter='\t')
    test_set_temp = list(temp)
test_set = []
for idx in range(len(test_set_temp)):
    test_set.append(list(map(float, test_set_temp[idx])))

# create X, y for train set
y_train = []
X_train = []
for item in train_set:
    if item[0] in classes2keep:
        y_train.append(item[0])
        X_train.append(item[1:])

y_train = [int(item-1) for item in y_train]
y_train_stats = Counter(y_train)

# create X, y for test set
y_test = []
X_test = []
for item in test_set:
    if item[0] in classes2keep:
        y_test.append(item[0])
        X_test.append(item[1:])

y_test = [int(item - 1) for item in y_test]
y_test_stats = Counter(y_test)
a = 0

classes2keep = [item - 1 for item in classes2keep]

X_train_norm = []
for item in X_train:
    X_train_norm.append(normalize(item))

X_test_norm = []
for item in X_test:
    X_test_norm.append(normalize(item))


for item in classes2keep:
    signal_X = []
    signal_y = []
    idx = 0
    count = 0
    while count < 16 and (idx < len(y_train) - 1):
        idx += 1
        if y_train[idx] == item:
            signal_X.append(X_train_norm[idx])
            signal_y.append(y_train[idx])
            count += 1
    grid_plot_save(n_row=4, n_col=4, signal=signal_X, labels=signal_y, path=myPath_save,
                               f_name='0_wgangp_train_samples_cl{}_norm.png'.format(item))

for item in classes2keep:
    signal_X = []
    signal_y = []
    idx = 0
    count = 0
    while count < 16 and (idx < len(y_test) - 1):
        idx += 1
        if y_test[idx] == item:
            signal_X.append(X_test_norm[idx])
            signal_y.append(y_test[idx])
            count += 1
    grid_plot_save(n_row=4, n_col=4, signal=signal_X, labels=signal_y , path=myPath_save,
                               f_name='0_wgangp_test_samples_cl{}_norm.png'.format(item))

myPath_save = "E:\\UTSA\\ECG_Synthesis\\dell_g7\\PycharmProjects\\paper2_gen_data\\" + d_set + \
               "multiclass\\genbeats_wgan_gp_cl_{}\\".format(classes2keep)
               
os.makedirs(myPath_save, exist_ok=True)
"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of ECG500 dataset %%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%%%  begin save train & test sets (X, y) %%%%%%%%%%%%%%%
'''
a = 0
with open(os.path.join(myPath_save, "X_train.json"), "w") as f:
    json.dump(X_train, f)
with open(os.path.join(myPath_save, "y_train.json"), "w") as f:
    json.dump(y_train, f)
with open(os.path.join(myPath_save, "X_test.json"), "w") as f:
    json.dump(X_test, f)
with open(os.path.join(myPath_save, "y_test.json"), "w") as f:
    json.dump(y_test, f)
'''
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%  end save train & test sets (X, y) %%%%%%%%%%%%%%%
dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# initialize gen and disc, note: discriminator should be called critic
gen = Gen_ac_wgan_gp_1d(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
critic = Disc_ac_wgan_gp_1d(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMG_SIZE).to(device)

initialize_weights_1d(gen)
initialize_weights_1d(critic)

# initialize optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

gen.train()
critic.train()

plt.close('all')
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(dataloader):
        real = real.unsqueeze(1).to(device)
        cur_batch_size = real.shape[0]
        labels = labels.type(torch.LongTensor).to(device)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((cur_batch_size, Z_DIM, 1)).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = gradient_penalty(critic, labels, real, fake, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            ### Train Generaor: max E[critic(gen_fake)] ↔ min -E[critic(gen_fake)]
            gen_fake = critic(fake, labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 200 == 0:
            now = datetime.datetime.now()
            print("{}".format(now.strftime("%d - %H:%M:%S")), end="      ")
            print(
                f"Epoch [{epoch:3d} / {NUM_EPOCHS:3d}]      Batch {batch_idx:4d}/{len(dataloader):5d} \
                     Loss D: {loss_critic: 6.4f},\tloss G: {loss_gen:6.4f}"
            )

            with torch.no_grad():
                fake = gen(noise, labels)
                # take out (up to) 32 examples

                plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.99, wspace=0.99)
                grid_plot_save(n_row=4, n_col=4, signal=fake.squeeze().to("cpu"), labels=labels,
                               all_classes=classes2keep, path=myPath_save,
                               f_name='wgangp_gb_ep{}_{}.png'.format(epoch, batch_idx))

            """
            # benchmarking Generator at end of each epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': gen.state_dict(),
                'optimizer_state_dict': opt_gen.state_dict(),
                'loss': loss_gen
            }, myPath_save + "generator_bench.tar")

            # benchmarking Discriminator end of each epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': critic.state_dict(),
                'optimizer_state_dict': opt_critic.state_dict(),
                'loss': loss_critic
            }, myPath_save + "discriminator_bench.tar")
            """

# save model
torch.save(gen.state_dict(), os.path.join(myPath_save, "generator_trained.pt"))
torch.save(critic.state_dict(), os.path.join(myPath_save, "discriminator_trained.pt"))

now = datetime.datetime.now()
laps = now - start_time
tot_sec = laps.total_seconds()
hr = int(tot_sec//3600)
min = int((tot_sec % 3600) // 60)
sec = tot_sec - (hr*3600 + min * 60)

print("\ntotal elapsed time {:02d}:{:02d}:{:02.2f}".format(hr, min, sec))

"""
#load model
model = model_name(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""

"""
Codes for:
YT, "Pytorch Conditional GAN Tutorial" by Aladdin Persson

https://www.youtube.com/watch?v=Hp-jWm2SzR8


"""
