import h5py
import numpy as np
from scipy.signal import correlate

mat_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_0to300s-using_spike2.mat'
edf_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/large_file_sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47.edf'

from sleep_tools.io import SleepRecording
from sleep_tools.analysis import SleepAnalyzer

rec = SleepRecording.from_edf(edf_path)
rec.raw.crop(tmax=300.0)
analyzer = SleepAnalyzer(rec)
features = analyzer.compute_all_features()

with h5py.File(mat_path, 'r') as mat:
    mat_emg_filt = mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EMGFilt/values'][0]
    mat_eeg_times = mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EMG_EMG/times'][0]

st_eeg_times = np.arange(len(features['eeg_filtered'])) / rec.sfreq
st_emg_filt = features['emg_filtered']
st_emg_filt_interp = np.interp(mat_eeg_times, st_eeg_times, st_emg_filt)

# find lag
cc = correlate(mat_emg_filt - np.mean(mat_emg_filt), st_emg_filt_interp - np.mean(st_emg_filt_interp), mode='full')
lags = np.arange(-len(mat_emg_filt)+1, len(mat_emg_filt))
best_lag = lags[np.argmax(cc)]
print(f"Best lag: {best_lag} samples (approx {best_lag / rec.sfreq:.3f} s)")
