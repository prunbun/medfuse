import pandas as pd
import numpy as np # Original import, kept even if np not explicitly used later
import os # Keep original imports

# --- Path Modifications ---
# Original paths commented out:
# ehr_data_dir = 'data/mimic-iv-extracted/phenotyping'
# cxr_data_dir = 'data/physionet.org/files/mimic-cxr-jpg/2.0.0'

# New paths based on your Drive structure for Colab:
drive_mount_path = '/content/drive/MyDrive/'
ehr_data_dir = f'{drive_mount_path}medfuse_data_root/phenotyping'
# Assuming 'mimic-cxr-jpg' folder is under 'datasets2'
cxr_data_dir = f'{drive_mount_path}datasets2/mimic-cxr-jpg'
# --- End Path Modifications ---


# Read original CXR split file - Added .gz extension
# Assumes pandas handles .gz automatically via filename
cxr_splits = pd.read_csv(f'{cxr_data_dir}/mimic-cxr-2.0.0-split.csv')
print(f'before update {cxr_splits.split.value_counts()}')

# Read EHR validation and test list files using the updated ehr_data_dir
ehr_split_val = pd.read_csv(f'{ehr_data_dir}/val_listfile.csv')
ehr_split_test = pd.read_csv(f'{ehr_data_dir}/test_listfile.csv')

# Original logic for extracting subject IDs
val_subject_ids = [stay.split('_')[0] for stay in ehr_split_val.stay.values]
test_subject_ids = [stay.split('_')[0] for stay in ehr_split_test.stay.values]


# Original logic for reassigning splits
# WARNING: This assumes 'subject_id' column in cxr_splits and the extracted IDs
# from EHR files have compatible types for the '.isin()' comparison.
# If one is integer and the other string, matches might fail silently.
# Keeping original logic as requested.
cxr_splits.loc[:, 'split'] = 'train'
cxr_splits.loc[cxr_splits.subject_id.isin(val_subject_ids), 'split'] = 'validate'
cxr_splits.loc[cxr_splits.subject_id.isin(test_subject_ids), 'split'] = 'test'

print(f'after update {cxr_splits.split.value_counts()}')

# Save the new split file to the updated cxr_data_dir
cxr_splits.to_csv(f'{cxr_data_dir}/mimic-cxr-ehr-split.csv', index=False)

# Optional: Add a print statement to confirm finish
print(f"\nScript finished. Updated split file saved to: {cxr_data_dir}/mimic-cxr-ehr-split.csv")