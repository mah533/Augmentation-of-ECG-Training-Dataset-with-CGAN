"""
visualizes beats, to check beats patterns visually
"""
import json
import os.path
import numpy as np
import torch
from matplotlib import pyplot as plt

from utils_ac_wgan_gp_ecg import grid_plot

classes = ['/', 'A', 'L', 'N', 'R', 'f', 'j']
cl = classes[0]

drive = "F:"
file_name = "genbeats_cl_{}.json".format(cl)
file_name_template = "template_cl_{}.json".format(cl)
folder = "MIT_BIH\\genbeats_wgan_gp_cl_{}".format(cl)
myPath_base = os.path.join(drive, "\\UTSA\\ECG_Synthesis\\dell_g7\\PycharmProjects\\paper2_gen_data")
myPath_read = os.path.join(myPath_base, folder)

with open(os.path.join(myPath_base, "MIT_BIH\\templates", file_name_template), "r") as f:
    templates_all = json.load(f)


with open(os.path.join(myPath_read, file_name), "r") as f:
    beats = json.load(f)
grid_plot(4, 4, templates_all[:16])

# num of plots
num = 100
"""
for idx in range(num):
    rand_vect = torch.randperm(len(beats))[:16]
    print(f"\nrand_vect: {rand_vect.tolist()}")
    signal = [beats[i] for i in rand_vect]
    grid_plot(4, 4, signal)
    plt.close()
"""
with open(os.path.join(myPath_read, file_name), "r") as f:
    genbeats = json.load(f)

a = 0

