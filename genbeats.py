"""
this file:
    *) uploads the trained generator (Cond. & Uncond. GAN models)
    *) generates beats (genbeats)
    *) produces sample plots
    *) screens genbeats (using DTW distance function, a template and a threshold)
"""
import copy
import datetime
import json
import os
import torch
from matplotlib import pyplot as plt
from model_ac_wgan_gp_ecg import Gen_ac_wgan_gp_1d
from model_wgan_gp_ecg import Gen_dcgan_gp_1d
import numpy as np
from collections import Counter
from utils_wgan_gp_ecg import grid_plot_save
from similaritymeasures import frechet_dist as frechet
from dtaidistance.dtw import distance as dtw

start_time = datetime.datetime.now()
print(("\n" + "*" * 50 + "\n\t\tstart time:      {0:02d}:{1:02d}:{2:02.0f}\n" + "*" * 50).format(
    start_time.hour, start_time.minute, start_time.second))

classes2keep = ['/', 'A', 'L', 'N', 'R', 'f', 'j']
classes2keep_folder = ['P', 'A', 'L', 'N', 'R', 'f', 'j']

cl = '/'        # for uncond. and cond


print("\nClass:                   {}".format(classes2keep))
print("Class (folder names):    {}\n".format(classes2keep_folder))

# %%%%%%%%%%%%% Cond/Uncond & Screened/Unscreened keys %%%%%%%%%%%%%%%%%%%%%%
# cond_key = "uncond"
cond_key = "cond"

screen_key = "screened"
# screen_key = "not_screened"

drive = "F:"

myPath_base = os.path.join(drive, "\\UTSA\\ECG_Synthesis\\dell_g7\\PycharmProjects\\paper2_gen_data\\MIT_BIH")

# unconditional GAN
# myPath_read = os.path.join(myPath_base, "01_{}\\generators_trained_uncond".format(cond_key))

# conditional GAN
myPath_read = os.path.join(myPath_base, "01_{}\\genbeats_ac_wgan_gp_{}_len_ratio_1".
                           format(cond_key, classes2keep_folder))

myPath_template = os.path.join(myPath_base, "templates")

# for uncond
# file_name_read = "generator_trained_cl_{}.pt".format(cl)
file_name_save = "gb_{}_{}_{}.json".format(cl, cond_key, screen_key)

# for cond
file_name_read = "generator_trained.pt"

# for Unconditional GAN
Z_DIM = 100
CHANNELS_IMG = 1
FEATURES_GEN = 64
BATCH_SIZE = 16

# for cond. GAN
IMG_SIZE = 256
NUM_CLASSES = len(classes2keep)
GEN_EMBEDDING = 100

n_rows = 4
n_cols = 4

myPath_save = os.path.join(myPath_base, "01_{}\\{}".format(cond_key, screen_key))
# %%%%%%%%%%%%%%%%%%    load model  %%%%%%%%%%%%%%%%%%%%%%%%
# Unconditional GAN
# model = Gen_dcgan_gp_1d(Z_DIM, CHANNELS_IMG, FEATURES_GEN)  # for uncond. GAN

# Conditional GAN
model = Gen_ac_wgan_gp_1d(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING)

model.load_state_dict(torch.load(os.path.join(myPath_read, file_name_read)))
model.eval()

# read templates
vals = [[] for _ in range(len(classes2keep_folder))]

templates_dict = dict(zip(classes2keep_folder, vals))

for cl_idx in classes2keep_folder:
    file_name_template = "template_cl_{}.json".format(cl_idx)
    with open(os.path.join(myPath_template, file_name_template)) as f:
        templates_dict[cl_idx] = json.load(f)

num_genbeats = 10000
count = 0
threshold = 2  # 2 for all class, 5 for class L, 2.3 for R
# _2 (additional genbeats) files use a different template and lower threshold
# threshold for class f & j reduced from 2 to 1.5

print("\n" + "%" * 25 + "\tclass: {}\t".format(cl) + "%" * 25)
labels = torch.tensor([classes2keep_folder.index(cl) for _ in range(16)])

# for uncond
# template = templates_dict[cl][0]

genbeats = []
while len(genbeats) <= num_genbeats:
    noise = torch.randn((BATCH_SIZE, Z_DIM, 1))         # for uncond. & cond. GAN

    # %%%%%%%%% for uncond. GAN & screened genbeats
    # screened = [model(noise).squeeze().tolist()[i] for i in range(16)
    #            if dtw(model(noise).squeeze().tolist()[i], template) <= threshold]

    # %%%%%%%%% for uncond. GAN & not screened genbeats
    # screened = [model(noise).squeeze().tolist()[i] for i in range(16)]

    # %%%%%%%%  for cond. GAN
    output = model(noise, labels).squeeze().tolist()
    # grid_plot(4, 4, output)
    # screened = [output[i] for i in range(16) if dtw(output[i], template) <= threshold]
    screened = [output[i] for i in range(16)]

    if len(screened) != 0:
        genbeats.extend(screened)
        count += 1
        if count % 200 == 0:
            now = datetime.datetime.now()
            print("{0:02d}:{1:02d}:{2:02.0f}".format(now.hour, now.minute, now.second), end=" ")
            print("\t\tcount: {:4d}, \t\tnum. of genbeats: {:5d}".format(count, len(genbeats)))

# for uncond & cond
with open(os.path.join(myPath_save, file_name_save), "w") as f:
    json.dump(genbeats, f)

a = 0

print("\nlength of genbeats: {}".format(len(genbeats)))

finish_time = datetime.datetime.now()
print(("\n\n" + "finish time = {0:02d}:{1:02d}:{2:02.0f}").format(
    finish_time.hour, finish_time.minute, finish_time.second))

laps = finish_time - start_time
tot_sec = laps.total_seconds()
h = int(tot_sec // 3600)
m = int((tot_sec % 3600) // 60)
s = int(tot_sec - (h * 3600 + m * 60))

print("total elapsed time = {:02d}:{:2d}:{:2d}".format(h, m, s))
print("total elapsed time (seconds) = {}".format(laps.total_seconds()))
