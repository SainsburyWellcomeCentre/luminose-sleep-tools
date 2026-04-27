import h5py
import numpy as np
from sleep_tools.io import SleepRecording
from sleep_tools.analysis import SleepAnalyzer

mat_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_0to300s-using_spike2.mat'
edf_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/large_file_sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47.edf'

rec = SleepRecording.from_edf(edf_path)
rec.raw.crop(tmax=300.0)
analyzer = SleepAnalyzer(rec)
features = analyzer.compute_all_features()

with h5py.File(mat_path, 'r') as mat:
    mat_emg_rms = mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EMGFilt/values'][0]
    mat_eeg_times = mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EMG_EMG/times'][0]
    
    mat_delta = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Delta_4001_/values'][0]
    mat_theta = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Theta_4001_/values'][0]
    mat_pow_times = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Delta_4001_/times'][0]

st_pow_times = features['times']
st_emg_rms = features['emg_rms']
st_emg_rms_interp = np.interp(mat_pow_times, st_pow_times, st_emg_rms)

# Note: Spike2 EMGFilt has length 153601 (same as EEG times) and is interpolated to sample rate?
# Let's check length of mat_emg_rms vs mat_pow_times
print("mat_emg_rms shape:", mat_emg_rms.shape)

st_emg_rms_full_interp = np.interp(mat_eeg_times, st_pow_times, st_emg_rms)
print(f"EMG RMS Corr: {np.corrcoef(mat_emg_rms, st_emg_rms_full_interp)[0,1]:.5f}")

st_delta = features['delta_power']
st_theta = features['theta_power']
st_delta_interp = np.interp(mat_pow_times, st_pow_times, st_delta)
st_theta_interp = np.interp(mat_pow_times, st_pow_times, st_theta)

print(f"Delta Corr: {np.corrcoef(mat_delta, st_delta_interp)[0,1]:.5f}")
print(f"Theta Corr: {np.corrcoef(mat_theta, st_theta_interp)[0,1]:.5f}")

from scipy.signal import correlate

cc = correlate(mat_delta - np.mean(mat_delta), st_delta_interp - np.mean(st_delta_interp), mode='full')
lags = np.arange(-len(mat_delta)+1, len(mat_delta))
best_lag = lags[np.argmax(cc)]
print(f"Best Delta lag: {best_lag} samples (approx {best_lag * 0.1:.3f} s)")

cc = correlate(mat_emg_rms - np.mean(mat_emg_rms), st_emg_rms_full_interp - np.mean(st_emg_rms_full_interp), mode='full')
lags = np.arange(-len(mat_emg_rms)+1, len(mat_emg_rms))
best_lag = lags[np.argmax(cc)]
print(f"Best EMG RMS lag: {best_lag} samples (approx {best_lag / rec.sfreq:.3f} s)")
