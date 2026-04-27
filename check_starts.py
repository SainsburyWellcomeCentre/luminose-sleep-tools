import h5py
import numpy as np

mat_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_0to300s-using_spike2.mat'

with h5py.File(mat_path, 'r') as mat:
    print("EEG start:", mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_EEG_EEG1A_B/start'][0][0])
    print("EEG offset:", mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_EEG_EEG1A_B/offset'][0][0])
    print("EMG start:", mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EMG_EMG/start'][0][0])
    print("EMGFilt start:", mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EMGFilt/start'][0][0])
    print("Delta start:", mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Delta_4001_/start'][0][0])
