# import pandas as pd
# import numpy as np # Original import, kept even if np not explicitly used later
# import os # Keep original imports
# import sys

# # --- Path Modifications ---
# # Original paths commented out:
# # ehr_data_dir = 'data/mimic-iv-extracted/phenotyping'
# # cxr_data_dir = 'data/physionet.org/files/mimic-cxr-jpg/2.0.0'

# # New paths based on your Drive structure for Colab:
# drive_mount_path = '/content/drive/MyDrive/'
# ehr_data_dir = f'{drive_mount_path}medfuse_data_root/phenotyping'
# # Assuming 'mimic-cxr-jpg' folder is under 'datasets2'
# cxr_data_dir = f'{drive_mount_path}datasets2'
# # --- End Path Modifications ---


# # Read original CXR split file - Added .gz extension
# # Assumes pandas handles .gz automatically via filename
# cxr_splits = pd.read_csv(f'{cxr_data_dir}/mimic-cxr-2.0.0-split.csv')
# print(f'before update {cxr_splits.split.value_counts()}')

# # Read EHR validation and test list files using the updated ehr_data_dir
# ehr_split_val = pd.read_csv(f'{ehr_data_dir}/val_listfile.csv')
# ehr_split_test = pd.read_csv(f'{ehr_data_dir}/test_listfile.csv')


# # Extract subject IDs from EHR lists (ensure they are strings)
# val_subject_ids = set([str(stay).split('_')[0] for stay in ehr_split_val['stay'].astype(str).values])
# test_subject_ids = set([str(stay).split('_')[0] for stay in ehr_split_test['stay'].astype(str).values])
# print(f"Extracted {len(val_subject_ids)} unique subject IDs (as strings) from EHR validation set.")
# print(f"Extracted {len(test_subject_ids)} unique subject IDs (as strings) from EHR test set.")

# # Convert CXR subject_id to string for reliable comparison
# # Handle potential NaN -> convert to int -> convert to str
# print("Converting CXR subject_id column to string format for comparison...")
# try:
#     # Fill NaN with a value that won't match (like -1), ensure integer, then string
#     cxr_splits['subject_id_str'] = cxr_splits['subject_id'].fillna(-1).astype(int).astype(str)
# except Exception as e:
#     print(f"ERROR: Failed to convert cxr_splits['subject_id'] to string: {e}")
#     # Optional: Print unique values to diagnose conversion issue
#     # print("Unique values in cxr_splits['subject_id']:", cxr_splits['subject_id'].unique())
#     sys.exit(1) # Exit if conversion fails

# print("Reassigning CXR splits based on EHR validation/test subjects...")
# cxr_splits['split'] = 'train' # Default to train

# # Perform comparison using the STRING version of subject_id
# cxr_splits.loc[cxr_splits['subject_id_str'].isin(val_subject_ids), 'split'] = 'validate'
# cxr_splits.loc[cxr_splits['subject_id_str'].isin(test_subject_ids), 'split'] = 'test'

# # Drop the temporary string column
# cxr_splits = cxr_splits.drop(columns=['subject_id_str'])

# print(f'\nafter update {cxr_splits.split.value_counts()}') # Should show non-zero val/test now

# # Save the new split file to the updated cxr_data_dir
# cxr_splits.to_csv(f'{cxr_data_dir}/mimic-cxr-ehr-split.csv', index=False)

# # Optional: Add a print statement to confirm finish
# print(f"\nScript finished. Updated split file saved to: {cxr_data_dir}/mimic-cxr-ehr-split.csv")


import pandas as pd
import os
import sys

print("--- Starting script to create CXR splits based ONLY on EHR assignments ---")

# --- Define Paths for Colab/Drive ---
drive_mount_path = '/content/drive/MyDrive/' # Base mount point

# Your specified paths relative to MyDrive
ehr_folder_rel = 'medfuse_data_root/phenotyping'
cxr_folder_rel = 'datasets2' # Contains the original split file

# Construct absolute paths within Colab's mounted Drive
ehr_data_dir = os.path.join(drive_mount_path, ehr_folder_rel)
cxr_data_dir = os.path.join(drive_mount_path, cxr_folder_rel)

print(f"EHR Data Directory: {ehr_data_dir}")
print(f"CXR Data Directory: {cxr_data_dir}")

# --- Define Input/Output Filenames ---
original_cxr_split_filename = 'mimic-cxr-2.0.0-split.csv'
# !!! CRITICAL ASSUMPTION: You MUST have a train_listfile.csv !!!
# If your training list file has a different name, change it here.
ehr_train_list_filename = 'train_listfile.csv' # <<< VERIFY/PROVIDE THIS FILE
ehr_val_list_filename = 'val_listfile.csv'
ehr_test_list_filename = 'test_listfile.csv'
# Output file containing only relevant CXR entries labeled by EHR split
output_split_filename = 'cxr_filtered_relabelled_by_ehr_split.csv' # More descriptive name

# Construct full file paths
original_cxr_split_path = os.path.join(cxr_data_dir, original_cxr_split_filename)
ehr_train_list_path = os.path.join(ehr_data_dir, ehr_train_list_filename)
ehr_val_list_path = os.path.join(ehr_data_dir, ehr_val_list_filename)
ehr_test_list_path = os.path.join(ehr_data_dir, ehr_test_list_filename)
output_split_path = os.path.join(cxr_data_dir, output_split_filename)
# --- End Path Definitions ---

# --- Function to load listfile and extract subject IDs as strings---
def load_ehr_subjects(filepath, split_name):
    """Loads a listfile, extracts unique subject IDs as strings."""
    print(f"Loading EHR {split_name} list from: {filepath}")
    try:
        df = pd.read_csv(filepath)
        if 'stay' not in df.columns:
             raise ValueError(f"'stay' column not found in {filepath}")
        # Extract subject IDs as strings, handling potential non-string data
        # Use set for immediate unique values
        subject_ids = set([str(stay).split('_')[0] for stay in df['stay'].astype(str).values])
        print(f" -> Loaded {len(df)} entries, found {len(subject_ids)} unique subject IDs.")
        if not subject_ids:
             print(f"WARNING: No subject IDs extracted from {filepath}. Check file content and format.")
        return subject_ids
    except FileNotFoundError:
        print(f"FATAL ERROR: File not found: {filepath}")
        # Exit if any required listfile is missing
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: Failed to load or process {filepath}: {e}")
        sys.exit(1)

# --- Load Data ---
print("\n--- Loading input files ---")
try:
    print(f"Loading original CXR splits from: {original_cxr_split_path}")
    cxr_splits_original = pd.read_csv(original_cxr_split_path)
    print(f" -> Loaded {len(cxr_splits_original)} original CXR split entries.")
    if 'subject_id' not in cxr_splits_original.columns:
        raise ValueError("'subject_id' column not found in CXR split file.")
     # Keep only essential columns to potentially save memory
    essential_cxr_cols = ['dicom_id', 'study_id', 'subject_id']
    cxr_splits_original = cxr_splits_original[essential_cxr_cols]
    print(f" -> Keeping columns: {essential_cxr_cols}")
except Exception as e:
    print(f"FATAL ERROR loading {original_cxr_split_path}: {e}")
    sys.exit(1)

# Load subjects from EHR lists
ehr_train_subjects = load_ehr_subjects(ehr_train_list_path, 'train')
ehr_val_subjects = load_ehr_subjects(ehr_val_list_path, 'validate')
ehr_test_subjects = load_ehr_subjects(ehr_test_list_path, 'test')

# Combine all unique subjects found in ANY EHR split list
all_ehr_subjects = ehr_train_subjects.union(ehr_val_subjects).union(ehr_test_subjects)
print(f"\nTotal unique subjects across all loaded EHR splits: {len(all_ehr_subjects)}")
if not all_ehr_subjects:
    print("FATAL ERROR: No subjects found in any EHR list files. Cannot proceed.")
    sys.exit(1)
# --- End Loading Data ---

# --- Filter and Relabel CXR Data ---
print("\n--- Filtering and Relabeling CXR data based on EHR subjects ---")

# Convert CXR subject_id to string once for efficient filtering/mapping
# Handle potential NaN -> convert to int -> convert to str
try:
    cxr_splits_original['subject_id_str'] = cxr_splits_original['subject_id'].fillna(-1).astype(int).astype(str)
except Exception as e:
    print(f"ERROR converting CXR subject_id to string: {e}")
    sys.exit(1)

# Filter CXR data: Keep only rows where subject_id exists in ANY loaded EHR split
cxr_splits_filtered = cxr_splits_original[cxr_splits_original['subject_id_str'].isin(all_ehr_subjects)].copy()
print(f"Filtered CXR entries: {len(cxr_splits_filtered)} (Kept only those whose subject_id is present in any loaded EHR list)")

# Define a function to assign the EHR split label based on priority: Test > Validate > Train
def assign_ehr_split(subject_id_str):
    if subject_id_str in ehr_test_subjects:
        return 'test'
    elif subject_id_str in ehr_val_subjects:
        return 'validate'
    elif subject_id_str in ehr_train_subjects: # Check train last
        return 'train'
    else:
        # This case should not happen due to the 'isin(all_ehr_subjects)' filter,
        # but is included as a safeguard.
        print(f"WARNING: Subject {subject_id_str} passed filter but not found in any split set?")
        return 'unknown'

# Apply the function to create the new split column
if not cxr_splits_filtered.empty:
    print("Assigning EHR split labels...")
    cxr_splits_filtered['split'] = cxr_splits_filtered['subject_id_str'].apply(assign_ehr_split)

    # Select and reorder columns for the final output
    # Keep essential IDs and the new 'split' column. Drop the temporary string id.
    final_df = cxr_splits_filtered[['dicom_id', 'study_id', 'subject_id', 'split']].copy()

    print("\nFinal split counts based purely on EHR assignments:")
    print(final_df['split'].value_counts())

    # --- Save Output ---
    print(f"\n--- Saving filtered and relabeled split file to: {output_split_path} ---")
    try:
        final_df.to_csv(output_split_path, index=False)
        print(f"Successfully saved new split file: {output_split_filename}")
    except Exception as e:
        print(f"ERROR: Failed to save new split file to {output_split_path}: {e}")
        sys.exit(1)
else:
    print("\nWARNING: No CXR entries found matching the subjects in the provided EHR lists. No output file created.")
# --- End Filter and Relabel ---

print("\n--- Script finished ---")