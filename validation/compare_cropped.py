import h5py
import numpy as np
from sleep_tools.io import SleepRecording
from sleep_tools.analysis import SleepAnalyzer
import mne

mat_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_0to300s-using_spike2.mat'
edf_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/large_file_sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47.edf'

print("Loading MAT...")
with h5py.File(mat_path, 'r') as mat:
    mat_delta = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Delta_4001_/values'][0]
    mat_theta = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Theta_4001_/values'][0]
    mat_td = mat['V03_ses_001_recording_001_0001_2026_03_25_16_13_47_T_D__4001_/values'][0]
    mat_pow_times = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Delta_4001_/times'][0]

print("Loading EDF and cropping to 300s...")
rec = SleepRecording.from_edf(edf_path)
rec.raw.crop(tmax=300.0)

print("Computing sleep-tools features...")
analyzer = SleepAnalyzer(rec)
features = analyzer.compute_all_features()

st_pow_times = features['times']
st_delta = features['delta_power']
st_theta = features['theta_power']
st_td = features['td_ratio']

st_delta_interp = np.interp(mat_pow_times, st_pow_times, st_delta)
st_theta_interp = np.interp(mat_pow_times, st_pow_times, st_theta)
st_td_interp = np.interp(mat_pow_times, st_pow_times, st_td)

print("\n=== Comparisons ===")
print(f"Delta Corr: {np.corrcoef(mat_delta, st_delta_interp)[0,1]:.5f}")
print(f"Theta Corr: {np.corrcoef(mat_theta, st_theta_interp)[0,1]:.5f}")
print(f"T:D Corr: {np.corrcoef(mat_td, st_td_interp)[0,1]:.5f}")

# Wait, we need to check how Spike2 calculates the T:D ratio
mat_manual_td = mat_theta / (mat_delta + 1e-12)
print(f"MAT T:D vs MAT Manual T:D Corr: {np.corrcoef(mat_td, mat_manual_td)[0,1]:.5f}")

st_manual_td = st_theta / (st_delta + 1e-12)
st_manual_td_interp = np.interp(mat_pow_times, st_pow_times, st_manual_td)
print(f"ST T:D vs ST Manual T:D Corr: {np.corrcoef(st_td_interp, st_manual_td_interp)[0,1]:.5f}")
