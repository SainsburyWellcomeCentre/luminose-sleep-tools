import h5py
import numpy as np

mat_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_0to300s-using_spike2.mat'

with h5py.File(mat_path, 'r') as mat:
    mat_emg_filt = mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EMGFilt/values'][0]
    mat_emg_raw = mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EMG_EMG/values'][0]

    print("EMG Filt min/max:", np.min(mat_emg_filt), np.max(mat_emg_filt))
    print("EMG Raw min/max:", np.min(mat_emg_raw), np.max(mat_emg_raw))
