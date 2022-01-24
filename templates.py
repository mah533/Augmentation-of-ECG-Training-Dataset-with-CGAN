"""
Codes for: YT, "Pytorch Conditional GAN Tutorial" by Aladdin Persson
"""
import copy
import datetime
import json
import os
from collections import Counter
from dtaidistance.dtw import distance as dtw
import matplotlib.pyplot as plt
import numpy as np
import torch
from ekg_class import dicts

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

num2descr, letter2num, letter2descr, num2letter = dicts()
start_time = datetime.datetime.now()
print(("\n" + "*" * 50 + "\n\t\tstart time:      {0:02d}:{1:02d}:{2:02.0f}\n" + "*" * 50).format(
    start_time.hour, start_time.minute, start_time.second))

drive = "F:"
myPath_base = os.path.join(drive, "\\UTSA\\ECG_Synthesis\\dell_g7")
myPath_dataset = os.path.join(myPath_base, "Datasets\\mitbih_datasets_Dictionaries")


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 16
CHANNELS_IMG = 1
CRITIC_ITERATIONS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
GEN_EMBEDDING = 100
LAMBDA_GP = 10
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
Z_DIM = 100
ratio = 1  # train/test split ratio


with open(os.path.join(myPath_dataset, "record_X_y_adapt_win_bef075_aft075_Normalized.json"), "r") as f:
    data = json.load(f)
stats_all_classes = Counter(np.asarray(data, dtype=object)[:, 2])

all_classes = ['/', 'A', 'E', 'F', 'J', 'L', 'N', 'Q', 'R', 'S', 'V', 'a', 'e', 'f', 'j']
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
"""
# %%%%%%%%%%%%%%%%%    end classes statistics    %%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%       begin MIT-BIH Dataset      %%%%%%%%%%%%%%%%%%%%%%%%%%%%

d_set = "MIT_BIH"
# Note: '/' and 'P' are the same

classes2keep = ['/', 'A', 'L', 'N', 'R', 'f', 'j']
classes2keep_folder = copy.copy(classes2keep)

if '/' in classes2keep_folder:
    idx = classes2keep_folder.index('/')
    classes2keep_folder.pop(idx)
    classes2keep_folder.insert(idx, 'P')

NUM_CLASSES = len(classes2keep)
IMG_SIZE = 256

# create dictionary of data to be kept
vals = [[] for _ in range(len(classes2keep))]
data2keep_dict = dict(zip(classes2keep, vals))

for item in data:
    if item[2] in classes2keep:
        data2keep_dict[item[2]].append(item[1])
del data

if '/' in classes2keep:
    temp = data2keep_dict['/']
    data2keep_dict.pop('/')
    data2keep_dict['P'] = temp

myPath_save = os.path.join(myPath_base, "PyCharmProjects\\paper2_data", d_set, "templates")
os.makedirs(myPath_save, exist_ok=True)


# plot samples of beats in classes2keep
"""
for cl in classes2keep_folder:
    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.suptitle("Class {} ({}: {}), count:{}".
                 format(classes2keep_folder.index(cl), cl, letter2descr[cl], len(data2keep_dict[cl])))

    count = 0
    for i in range(3):
        for j in range(3):
            count += 1
            if count >= len(data2keep_dict[cl]):
                continue
            axes[i][j].plot(data2keep_dict[cl][count])
            axes[i][j].grid()
    plt.savefig(os.path.join(myPath_save, "00_sample_cl{}".format(classes2keep_folder.index(cl))))

plt.close("all")
"""

# recall: classes2keep = ['/', 'A', 'L', 'N', 'R', 'f', 'j']
cl = "A"
template_folder = "templates"
f_name = "templates_{}_.json".format(cl)
num_templates = 50

templates_all = data2keep_dict[cl]
# %%%%%%%%%%%%%%%%%%%%%%%%% begin save templates %%%%%%%%%%%%%%%%%%%%%%%
templates = templates_all[:num_templates]

with open((os.path.join(myPath_save, f_name)), "w") as f:
    json.dump(templates, f)
a = 0
"""
# %%%%%%%%%%%%%%%%%%%%%%%%% end save templates %%%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%%% begin filter templates %%%%%%%%%%%%%%%%%%%%%%%

# templates are screened here based on their distance from a visually selected template
# which seemed similar to this:

# https://www.healio.com/cardiology/learn-the-heart/ecg-review/ecg-topic-reviews-and-criteria/premature-ventricular-contractions-review
threshold = 2
f_name = "template_{}_screened.json".format(cl)
os.makedirs(os.path.join(path_class, template_folder), exist_ok=True)
templates_screened = []
for idx, item in enumerate(templates_all):
    if dtw(template, item) <= threshold:
        templates_screened.append(item)

with open((os.path.join(path_class, template_folder, f_name)), "w") as f:
    json.dump(templates_screened, f)
"""

