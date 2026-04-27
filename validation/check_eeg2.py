import h5py
import numpy as np
from sleep_tools.io import SleepRecording
from sleep_tools.analysis import SleepAnalyzer

mat_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_0to300s-using_spike2.mat'
edf_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/large_file_sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47.edf'

rec = SleepRecording.from_edf(edf_path)
rec.raw.crop(tmax=300.0)

with h5py.File(mat_path, 'r') as mat:
    mat_delta = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Delta_4001_/values'][0]
    mat_theta = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Theta_4001_/values'][0]
    mat_pow_times = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Delta_4001_/times'][0]
    
    mat_eeg_filt = mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EEGFilt/values'][0]
    mat_eeg_times = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_EEG_EEG1A_B/times'][0]

analyzer = SleepAnalyzer(rec)

features_eeg2 = analyzer.compute_all_features(eeg_channel='EEG2', fft_size=512)
st_eeg2_filt = features_eeg2['eeg_filtered']
st_eeg_times_st = np.arange(len(st_eeg2_filt)) / rec.sfreq

st_eeg2_filt_interp = np.interp(mat_eeg_times, st_eeg_times_st, st_eeg2_filt)
print(f"EEG Filt Corr with EEG2: {np.corrcoef(mat_eeg_filt, st_eeg2_filt_interp)[0,1]:.5f}")

st_delta2 = features_eeg2['delta_power']
st_delta2_interp = np.interp(mat_pow_times, features_eeg2['times'], st_delta2)

from scipy.signal import correlate

cc = correlate(mat_delta - np.mean(mat_delta), st_delta2_interp - np.mean(st_delta2_interp), mode='full')
lags = np.arange(-len(mat_delta)+1, len(mat_delta))
best_lag = lags[np.argmax(cc)]
print(f"Best Delta lag with EEG2: {best_lag} samples (approx {best_lag * 0.1:.3f} s)")

# Wait, let's also try EEG2 raw correlation, just in case "EEGFilt" was indeed from EEG2
st_eeg2_raw = rec.raw.get_data(picks=['EEG2'])[0]
st_eeg2_raw_interp = np.interp(mat_eeg_times, st_eeg_times_st, st_eeg2_raw)
mat_eeg_raw = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_EEG_EEG1A_B/values'][0]
print(f"EEG Raw Corr with EEG2: {np.corrcoef(mat_eeg_raw, st_eeg2_raw_interp)[0,1]:.5f}")

