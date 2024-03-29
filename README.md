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

download the MITBIH dataset file ("**_record_X_y_adapt_win_bef075_aft075_Normalized.json_**") from Google Drive via the following link and place it in the proper folder accordingly:

https://drive.google.com/file/d/1d2gUuhJeWwVtKfzPbZx9gJgeBhXp3ibb/view?usp=sharing

raw signals (from PhysioNet website https://physionet.org/content/mitdb/1.0.0/) are segmented according to an adaptive window scheme: heart rate (R-R distance) is calculated at each beat and 75% of the distance before and after each R-peak are used as the cutoffs for segmentations. The individual beats are put in a dictionary. The _keys_ are the record numbers and the _values_ are lists of segmented beats. 

link to all data in MITBIH Arhythmia dataset in 'dictionary' format: https://drive.google.com/file/d/1mVzEMRCzA-j3CFEgOvqGSQcQAbg0Gz4N/view?usp=drive_link

link to the 7 classes of real data used in this study: https://drive.google.com/file/d/1h0TtJAvJJK3IwDbqzK6pLKVBQw5VpvDL/view?usp=drive_link

