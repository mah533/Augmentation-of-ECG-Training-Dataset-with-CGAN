**_paper_**: Arrhythmia Classification using CGAN-augmented ECG Signals

**_arXiv_**: https://arxiv.org/abs/2202.00569

**_please cite as_**:       
@article{adib2022arrhythmia,       
  title={Arrhythmia Classification using CGAN-augmented ECG Signals},  
  author={Adib, Edmond and Afghah, Fatemeh and Prevost, John J},  
  journal={arXiv preprint arXiv:2202.00569},  
  year={2022}  
}

the read/save paths in the file ("**_main_ac_wgan_gp_ecg.py_**") should adjusted according to your filing system

download the MITBIH dataset file ("**_record_X_y_adapt_win_bef075_aft075_Normalized.json_**") from Google Drive via the following link nd place it in the proper folder accordingly:

https://drive.google.com/file/d/1d2gUuhJeWwVtKfzPbZx9gJgeBhXp3ibb/view?usp=sharing

signals are segmented according to an adaptive window scheme: heart rate (R-R distance) is calculated for each beat and 75% of the distance before and after each R-peak are used as cutoffs for segmentations. The individual beats are put in a dictionary. The keys are record numbers and the values are the segmented beats. 

