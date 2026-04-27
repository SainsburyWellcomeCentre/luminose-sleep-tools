import h5py
import numpy as np

mat_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_0to300s-using_spike2.mat'
h5_path = 'example_data/comparison_with_spike2_to_debug_inconsistencies/large_file_sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_using_sleeptools.h5'

with h5py.File(mat_path, 'r') as mat:
    print("MAT file structure:")
    def print_mat(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f" {name}: {obj.shape}, {obj.dtype}")
    mat.visititems(print_mat)

with h5py.File(h5_path, 'r') as h5:
    print("\nH5 file structure:")
    def print_h5(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f" {name}: {obj.shape}, {obj.dtype}")
    h5.visititems(print_h5)
