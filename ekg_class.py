import json
import matplotlib.pyplot as plt
import numpy as np
import cmath
from scipy import signal
from scipy.signal import resample
import wfdb
import collections
import pickle


'''
import pandas as pd
import os
import shutil
from biosppy import tools as st
import copy
'''


def codes():
    """
    matplotlib line codes are read from (n1 line codes): plt_line_code.txt
    matplotlib colors are read from (n2 colors): plt_colors.txt

    and n1*n2 codes are made


    return: graphing codes (a list, n1*n2)
    """

    file_line_codes = 'plt_line_codes.txt'
    # file_colors = 'plt_colors.txt'

    colors = list(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    # data = open('').read('file_line_codes')
    with open(file_line_codes, 'rb') as f:
        data = str(f.read())

    line_codes = data.split('\\r\\n')
    line_codes[0] = '-'
    line_codes[-1] = '_'

    code = []
    for i in range(len(line_codes)):
        j = 0
        while True:
            if j >= len(colors):
                break
            a = line_codes[i] + colors[j]
            code.append(a.replace(" ", ""))
            j += 1
    return code


def dicts():
    """
    # Create annotation  dictionary from WFDB Library


    :returns: 4 dictionaries
    """

    with open('label_descr.txt', 'rb') as f:
        label_descr_list = pickle.load(f)
    '''
    label_descr_list.pop(label_descr_list.index(['/', 'Paced beat']))
    label_descr_list.append((['P', 'Paced beat']))
    with open('label_descr.txt', "wb") as file:
        pickle.dump(label_descr_list, file)
    '''
    # building graphing codes

    # build lists of labels & descriptions from WFDB library
    annot_class_alpha   = [label_descr_list[i][0] for i in range(len(label_descr_list))]
    annot_class_descr   = [label_descr_list[i][1] for i in range(len(label_descr_list))]
    annot_class_num     = list(range(len(label_descr_list)))

    # Create annotation  dictionary from WFDB Library
    dumzip_obj = zip(annot_class_num, annot_class_descr)
    num2descr = dict(dumzip_obj)

    # create annotation dictionary for classification purposes
    dum_zip_obj = zip(annot_class_alpha, annot_class_num)
    letter2num  = dict(dum_zip_obj)

    # create annotation dictionary for classification purposes
    dum_zip_obj  = zip(annot_class_alpha, annot_class_descr)
    letter2descr = dict(dum_zip_obj)

    num2letter = dict(zip(letter2num.values(), letter2num.keys()))

    return num2descr, letter2num, letter2descr, num2letter


class EKG:
    """
    ekg is a class of variables. It's attributes and methods should cover all the information required

    """
    def __init__(self, path, file_name, atr_file_ext, sample_from=0, sample_to=650000):
        record_ttl = wfdb.rdrecord(path + '\\' + file_name)
        annots_ttl = wfdb.rdann(path + '\\' + file_name, atr_file_ext)

        self.num_samples_ttl  = record_ttl.p_signal.shape[0]
        self.num_channels     = record_ttl.p_signal.shape[1]

        self.chan1 = list(record_ttl.p_signal[:, 0][sample_from:sample_to])
        self.chan2 = list(record_ttl.p_signal[:, 1][sample_from:sample_to])

        # read annotations from   ...   to   ...
        self.annot_symb_orig_ttl = annots_ttl.symbol
        self.annot_idx_orig_ttl  = annots_ttl.sample

        self.annot_symb_orig = annots_ttl.symbol[sample_from:sample_to]
        self.annot_idx_orig  = annots_ttl.sample[sample_from:sample_to]

        # the whole record
        # removes unwanted symbols & their corresp. indices
        symb_cor_ttl        = []
        peak_indices_cor_ttl = []

        for i in range(len(annots_ttl.symbol)):
            if not (annots_ttl.symbol[i] == '+' or annots_ttl.symbol[i] == '~' or
                    annots_ttl.symbol[i] == '[' or annots_ttl.symbol[i] == ']' or
                    annots_ttl.symbol[i] == '|'):
                symb_cor_ttl.append(annots_ttl.symbol[i])
                peak_indices_cor_ttl.append(annots_ttl.sample[i])

        self.annot_symb_cor_ttl = symb_cor_ttl
        self.annot_idx_cor_ttl  = peak_indices_cor_ttl

        self.annot_symb_cor = []      # self.annot_symb_cor_ttl[sample_from:sample_to]
        self.annot_idx_cor  = []      # self.annot_idx_cor_ttl [sample_from:sample_to]

        for i in range(len(self.annot_symb_cor_ttl)):
            if self.annot_idx_cor_ttl[i] > sample_to:
                break

            if (self.annot_idx_cor_ttl[i] >= sample_from) & (self.annot_idx_cor_ttl[i] <= sample_to):
                self.annot_symb_cor.append(self.annot_symb_cor_ttl[i])
                self.annot_idx_cor.append(self.annot_idx_cor_ttl[i])

        self.classes_orig_stats_ttl = collections.Counter(self.annot_symb_orig_ttl)
        self.classes_cor_stats_ttl  = collections.Counter(self.annot_symb_cor_ttl)

        self.classes_orig_stats = collections.Counter(self.annot_symb_orig)
        self.classes_cor_stats  = collections.Counter(self.annot_symb_cor)

        '''
        ******************************************************************
                                     Waveforms 
                (segmentation based on corrected annotation indices)  
        ******************************************************************   
        '''
        self.waveforms = []
        # idx_adj = 1
        thresh = 10
        for i in range(1, len(self.annot_idx_cor)):
            i_from = self.annot_idx_cor[i-1]
            i_to   = self.annot_idx_cor[i]

            # ***************************************************************************************************
            # ****************************   double check these 4 following lines   *****************************
            # ***************************************************************************************************
            corr1 = np.argmax(list(record_ttl.p_signal[:, 0][i_from - thresh:i_from + thresh]))
            corr2 = np.argmax(list(record_ttl.p_signal[:, 0][i_to - thresh:i_to + thresh]))

            wf = list(self.chan1[i_from - thresh + corr1:i_to - thresh + corr2])
            self.waveforms.append(wf)
            # plt.plot(wf)
        # plt.show()
        self.num_waveforms  = len(self.waveforms)
        self.waveforms_lens = [len(self.waveforms[i])   for i in range(self.num_waveforms)]
        # self.resamp_num    = int((max(self.waveforms_lens) + min(self.waveforms_lens)) / 2)
        self.resamp_num     = int(sum(self.waveforms_lens)/len(self.waveforms_lens))

        '''
        ******************************************************************
                                  Resampling & FFT   
        ******************************************************************   
        '''

        self.waveforms_resamped                 = []
        self.waveforms_resamped_fft_rect        = []
        self.waveforms_resamped_fft_polar_mag   = []
        self.waveforms_resamped_fft_polar_angle = []

        for i in range(self.num_waveforms):
            wf_resamp = resample(self.waveforms[i], self.resamp_num)
            self.waveforms_resamped.append(wf_resamp)
            wf_fft = np.fft.fft(wf_resamp)
            self.waveforms_resamped_fft_rect.append(wf_fft)
            wf_fft_polar       = [list(cmath.polar(wf_fft[i]))    for i in range(len(wf_fft))]
            wf_fft_polar_mag   = [row[0]      for row in wf_fft_polar]
            wf_fft_polar_angle = [row[1]      for row in wf_fft_polar]
            self.waveforms_resamped_fft_polar_mag.append(wf_fft_polar_mag)
            self.waveforms_resamped_fft_polar_angle.append(wf_fft_polar_angle)
            '''
            ******************************************************************
                                  Amplitudal Normalization  
            ******************************************************************   
            '''
        self.waveforms_resamped_normalized = []

        for i in range(self.num_waveforms):
            y_min = min(self.waveforms_resamped[i])
            y_0 = self.waveforms_resamped[i][0]
            temp = []
            for j in range(len(self.waveforms_resamped[i])):
                y_j  = self.waveforms_resamped[i][j]
                if y_0 == y_min:
                    appendee = 0
                else:
                    appendee = (y_j - y_min)/(y_0 - y_min)
                temp.append(appendee)
            self.waveforms_resamped_normalized.append(temp)

            '''
            ******************************************************************
                            Plotting Amplitudally Normalized Waveforms  
            ******************************************************************   
            '''
            '''
        n = list(range(self.resamp_num))
        n=[n[i]/self.resamp_num     for i in range(len(n))]
        for i in range(len(self.waveforms_resamped_normalized)):
            fig = plt.figure("Waveform: {} (of {}),    Original Length: {},    Resampled Length: {}".
                             format(i, self.num_waveforms,   self.waveforms_lens[i], self.resamp_num),
                             figsize=(7,6), tight_layout="T")


            ax1 = fig.add_subplot(2,1,1)
            ax1.plot(n, self.waveforms_resamped[i])
            ax1.set_title("Waveform (temporally normalized)")
            ax1.grid()

            ax2 = fig.add_subplot(2,1,2)
            ax2.plot(n, self.waveforms_resamped_normalized[i])
            ax2.set_title("Amplitudally Normalized Waveform (temporally normalized)")
            ax2.grid()

            ax1.axhline(linewidth=1.3, y=0, color='g', dashes =(6,4))
            ax2.axhline(linewidth=1.3, y=0, color='g', dashes =(6,4))

            plt.show()

        a = 0

        '''

    @classmethod
    def plot(cls, sig, n_from, n_to, fs, title=None):
        t_from_sec = (n_from/fs)
        t_to_sec   = (n_to/fs)
        step = 1/fs
        t = np.arange(t_from_sec, t_to_sec, step)
        if len(t) < len(sig):
            sig = sig[0:len(t)]
        elif len(t) > len(sig):
            t = t[0:len(sig)]

        plt.title(title)
        plt.plot(t, sig)
        plt.xlabel('Time (sec)')
        plt.grid(b='True', which='major', linestyle='-', linewidth='1.1')
        plt.grid(b='True', which='minor', linestyle=':')
        plt.minorticks_on()
        # plt.show()

    @classmethod
    def derivative1(cls, sig):
        # calculates the first derivative of signal by central difference
        # slope = arithmetic average between slopes at points (n-1, n+1) & (n-2, n+2)
        der = []
        for n in range(len(sig)-3):
            if n < 2:
                der.append(0)
                continue
            dum = -sig[n-2]-2*sig[n-1]+2*sig[n+1]+sig[n+2]
            der.append(dum)
        return der

    @classmethod
    def decimal_time_2_hms(cls, dtime_in_seconds):
        hours = dtime_in_seconds/3600
        minutes = 60 * (hours   % 1)
        seconds = 60 * (minutes % 1)

        hours_s   = str(int(hours))
        if int(hours_s) < 10:
            hours_s = '0' + hours_s
        minutes_s = str(int(minutes))
        if int(minutes) < 10:
            minutes_s = '0' + minutes_s
        seconds_s = "{:3.1f}".format(seconds)
        if int(seconds) < 10:
            seconds_s = '0' + seconds_s

        hms_format = hours_s + ":" + minutes_s + ":" + seconds_s
        return hms_format

    # function to return key for any value in a Dictionary
    @classmethod
    def get_key(cls, my_dict, val):
        for key, value in my_dict.items():
            if val == value:
                return key

        return "key doesn\'t exist"
