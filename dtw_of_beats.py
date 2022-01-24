"""
calculates average DTW distance of beats (generated or real)
from a visually selected template
"""
import json
import os
import sys
from datetime import datetime
from statistics import mean

import numpy as np
from similaritymeasures import frechet_dist as frechet
from dtaidistance.dtw import distance as dtw

start_time = datetime.now()
print(("\n" + "*" * 61 + "\n\t\t\t\t\tstart time  {0:02d}:{1:02d}:{2:02.0f}\n" + "*" * 61).format(
    start_time.hour, start_time.minute, start_time.second))

aug_key = "augmented"
# aug_key = ""

cond_key = "uncond"
# cond_key = "cond"

screen_key = "screened"
# screen_key = "not_screened"

drive = "F:"
myPath_base = os.path.join(drive, "\\UTSA\\ECG_Synthesis\\dell_g7\\PycharmProjects")

aux_path_2 = "paper2_data\\MIT_BIH\\gb_{}\\{}".format(cond_key, screen_key)
# aux_path_3 ="paper3_data\\gb_{}\\gb_{}_{}".format(cond_key, screen_key)
# f_name_3 = "data_gb_dict_{}_{}.json"

# recall: classes2keep = ['P', 'A', 'L', 'N', 'R', 'f', 'j']

cl = 'j'
# print('class: {}, {}, {}'.format(cl, cond_key, screen_key))
print('real data, class: {}'.format(cl))
f_name_2 = "gb_{}_{}_{}.json".format(cl, cond_key, screen_key)

myPath_template = os.path.join(myPath_base, "paper2_data\\MIT_BIH\\templates")
f_name_template = "template_cl_{}.json".format(cl)

myPath_read_gb = os.path.join(myPath_base, aux_path_2)
myPath_read_rl = os.path.join(os.path.join(drive, "\\UTSA\\ECG_Synthesis\\dell_g7\\Datasets"))

with open(os.path.join(myPath_read_rl, "data_rl_dict.json"), 'r') as f:
    data_rl_dict = json.load(f)

# with open(os.path.join(myPath_read_gb, f_name_2), 'r') as f:
#    beats_all = json.load(f)

# beats = beats_all[:5000]
with open(os.path.join(myPath_template, f_name_template), 'r') as f:
    templates = json.load(f)

template_num = 3
template = templates[template_num]

if len(data_rl_dict[cl]) <= 5000:
    beats = data_rl_dict[cl]
else:
    beats = data_rl_dict[cl][:5000]

dist_dtw = []
# dist_frechet = []
for idx, beat in enumerate(beats):
    dist_dtw.append(dtw(beat, template))
    # dist_frechet.append(frechet(beat, template))

dist_dtw_mean = mean(dist_dtw)
# dist_frechet_mean = mean(dist_frechet)


original_stdout = sys.stdout
# %%%%%%%%%%%%%%%%%%%%%%% begin: write to file (genbeats) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
with open(os.path.join(myPath_read_gb, 'dist_{}_{}.txt'.format(cond_key, screen_key)), 'a') as f:
    sys.stdout = f
    print('\nNumber of beats in the class {}: {}\n'.format(cl, len(beats)))
    print('Average DTW distance of genbeats ({}, {}) from template # {}:'.format(cond_key, screen_key, template_num))
    print('\n\tcl: {}\n\tave DTW: {:5.3f}'.format(cl, dist_dtw_mean))
    print('%' * 60)

    # print('\nAverage Frechet distance of genbeats ({}, {}:)'.format(cond_key, screen_key))
    # print('\t ave Frechet: {:5.3f}'.format(dist_frechet_mean))
"""
# %%%%%%%%%%%%%%%%%%%%%%% end: write to file (genbeats) %%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%% begin: write to file (data_rl_dict) %%%%%%%%%%%%%%%%%%%%%%%%%%
with open(os.path.join(myPath_base, "paper2_data\\final_report_files", "dist_real_beats.txt"), 'a') as f:
    sys.stdout = f
    print('\nNumber of beats in the class {}: {}\n'.format(cl, len(beats)))
    print('Average DTW distance of real beats from template # {}:'.format(template_num))
    print('\n\tcl: {}\n\tave DTW: {:5.3f}'.format(cl, dist_dtw_mean))
    print('%' * 60)


# %%%%%%%%%%%%%%%%%%%%%%% end: write to file (data_rl_dict) %%%%%%%%%%%%%%%%%%%%%%%%%%%%

sys.stdout = original_stdout

finish_time = datetime.now()
print(("\n\n\n" + "finish time = {0:02d}:{1:02d}:{2:02.0f}").format(
    finish_time.hour, finish_time.minute, finish_time.second))

laps = finish_time - start_time
tot_sec = laps.total_seconds()
h = int(tot_sec // 3600)
m = int((tot_sec % 3600) // 60)
s = int(tot_sec - (h * 3600 + m * 60))

print("total elapsed time = {:02d}:{:02d}:{:02d}".format(h, m, s))

a = 0
