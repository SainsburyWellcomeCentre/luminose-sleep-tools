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

analyzer = SleepAnalyzer(rec)

# Try EEG1
features_eeg1 = analyzer.compute_all_features(eeg_channel='EEG1')
st_delta1 = features_eeg1['delta_power']
st_delta_interp1 = np.interp(mat_pow_times, features_eeg1['times'], st_delta1)
corr1 = np.corrcoef(mat_delta, st_delta_interp1)[0,1]

# Clear cache and Try EEG2
analyzer.invalidate_cache()
features_eeg2 = analyzer.compute_all_features(eeg_channel='EEG2')
st_delta2 = features_eeg2['delta_power']
st_delta_interp2 = np.interp(mat_pow_times, features_eeg2['times'], st_delta2)
corr2 = np.corrcoef(mat_delta, st_delta_interp2)[0,1]

print(f"Delta Corr with EEG1: {corr1:.5f}")
print(f"Delta Corr with EEG2: {corr2:.5f}")

# Try increasing FFT size to 1024 or 512 for EEG1 and EEG2 to match Spike2
analyzer.invalidate_cache()
features_eeg1_fft = analyzer.compute_all_features(eeg_channel='EEG1', fft_size=512)
st_delta1_fft = np.interp(mat_pow_times, features_eeg1_fft['times'], features_eeg1_fft['delta_power'])
corr1_fft = np.corrcoef(mat_delta, st_delta1_fft)[0,1]

analyzer.invalidate_cache()
features_eeg2_fft = analyzer.compute_all_features(eeg_channel='EEG2', fft_size=512)
st_delta2_fft = np.interp(mat_pow_times, features_eeg2_fft['times'], features_eeg2_fft['delta_power'])
corr2_fft = np.corrcoef(mat_delta, st_delta2_fft)[0,1]

print(f"Delta Corr with EEG1 (fft_size=512): {corr1_fft:.5f}")
print(f"Delta Corr with EEG2 (fft_size=512): {corr2_fft:.5f}")
