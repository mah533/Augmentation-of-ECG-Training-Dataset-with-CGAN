Paper: Arrhythmia Classification using CGAN-augmented ECG Signals

arXiv: https://arxiv.org/abs/2202.00569

please cite as:
@article{adib2022arrhythmia,       
  title={Arrhythmia Classification using CGAN-augmented ECG Signals},  
  author={Adib, Edmond and Afghah, Fatemeh and Prevost, John J},  
  journal={arXiv preprint arXiv:2202.00569},  
  year={2022}  
}

the read/save paths in the file ("main_ac_wgan_gp_ecg.py") should adjusted according to your filing system

link to the MITBIH dataset file ("record_X_y_adapt_win_bef075_aft075_Normalized.json") on Google Drive:

https://drive.google.com/file/d/1d2gUuhJeWwVtKfzPbZx9gJgeBhXp3ibb/view?usp=sharing

signals are segmented according to an adaptive window scheme: heart rate (R-R distance) is calculated for each beat and 75% of the distance before and after each R-peak are used as cutoffs for segmentations. The individual beats are put in a dictionary. The keys are record numbers and the values are the segmented beats. 

