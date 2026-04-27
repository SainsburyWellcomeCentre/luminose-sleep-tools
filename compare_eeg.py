import h5py
import numpy as np
from sleep_tools.io import SleepRecording
from sleep_tools.analysis import SleepAnalyzer

mat_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_0to300s-using_spike2.mat'
edf_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/large_file_sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47.edf'

print("Loading MAT...")
with h5py.File(mat_path, 'r') as mat:
    mat_eeg_raw = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_EEG_EEG1A_B/values'][0]
    mat_eeg_filt = mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EEGFilt/values'][0]
    mat_eeg_times = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_EEG_EEG1A_B/times'][0]

print("Loading EDF and cropping to 300s...")
rec = SleepRecording.from_edf(edf_path)
rec.raw.crop(tmax=300.0)

print("Computing sleep-tools features...")
analyzer = SleepAnalyzer(rec)
features = analyzer.compute_all_features()

st_eeg_times = np.arange(len(features['eeg_filtered'])) / rec.sfreq
st_eeg_filt = features['eeg_filtered']

# EDF raw
st_eeg_raw = rec.raw.get_data(picks=['EEG1'])[0]

st_eeg_raw_interp = np.interp(mat_eeg_times, st_eeg_times, st_eeg_raw)
st_eeg_filt_interp = np.interp(mat_eeg_times, st_eeg_times, st_eeg_filt)

print("\n=== EEG Comparisons ===")
print(f"EEG Raw Corr: {np.corrcoef(mat_eeg_raw, st_eeg_raw_interp)[0,1]:.5f}")
print(f"EEG Filt Corr: {np.corrcoef(mat_eeg_filt, st_eeg_filt_interp)[0,1]:.5f}")
