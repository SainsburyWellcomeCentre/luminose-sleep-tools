import h5py
import numpy as np

mat_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_0to300s-using_spike2.mat'
h5_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/large_file_sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_using_sleeptools.h5'

# Load MAT
with h5py.File(mat_path, 'r') as mat:
    mat_eeg_raw = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_EEG_EEG1A_B/values'][0]
    mat_eeg_filt = mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EEGFilt/values'][0]
    mat_emg_raw = mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EMG_EMG/values'][0]
    mat_emg_filt = mat['b_003_ses_001_recording_001_0001_2026_03_25_16_13_47_EMGFilt/values'][0]
    mat_delta = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Delta_4001_/values'][0]
    mat_theta = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Theta_4001_/values'][0]
    mat_td = mat['V03_ses_001_recording_001_0001_2026_03_25_16_13_47_T_D__4001_/values'][0]

    # Time axes
    mat_eeg_times = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_EEG_EEG1A_B/times'][0]
    mat_pow_times = mat['V3_ses_001_recording_001_0001_2026_03_25_16_13_47_Delta_4001_/times'][0]

print(f"MAT loaded. EEG points: {len(mat_eeg_raw)}, Power points: {len(mat_delta)}")

# Load H5 and get sleep-tools features
with h5py.File(h5_path, 'r') as h5:
    st_eeg_times = np.arange(len(h5['signals/EEG1'])) / 512.0 # assuming 512 Hz
    st_eeg_filt = h5['signals/EEG1'][:] # wait, is it filtered in h5? No, h5 typically has raw signals. Wait, I should check if the signals are filtered.
    
    st_pow_times = h5['epochs/times'][:]
    st_delta = h5['epochs/features/delta_power'][:]
    st_theta = h5['epochs/features/theta_power'][:]
    st_td = h5['epochs/features/td_ratio'][:]

# The H5 power times might not exactly match MAT power times since H5 might span longer or start at 0.
st_delta_interp = np.interp(mat_pow_times, st_pow_times, st_delta)
st_theta_interp = np.interp(mat_pow_times, st_pow_times, st_theta)
st_td_interp = np.interp(mat_pow_times, st_pow_times, st_td)

print("\n=== Comparisons ===")

# calculate correlation coefficients
print(f"Delta Corr: {np.corrcoef(mat_delta, st_delta_interp)[0,1]:.5f}")
print(f"Theta Corr: {np.corrcoef(mat_theta, st_theta_interp)[0,1]:.5f}")
print(f"T:D Corr: {np.corrcoef(mat_td, st_td_interp)[0,1]:.5f}")

# Check T:D ratio manually in MAT
mat_manual_td = mat_theta / (mat_delta + 1e-12)
print(f"MAT T:D vs MAT Manual T:D Corr: {np.corrcoef(mat_td, mat_manual_td)[0,1]:.5f}")
print(f"MAT T:D Mean: {np.mean(mat_td):.5f}, MAT Manual T:D Mean: {np.mean(mat_manual_td):.5f}")

# Check T:D ratio manually in ST
st_manual_td = st_theta / (st_delta + 1e-12)
st_manual_td_interp = np.interp(mat_pow_times, st_pow_times, st_manual_td)
print(f"ST T:D vs ST Manual T:D Corr: {np.corrcoef(st_td_interp, st_manual_td_interp)[0,1]:.5f}")

