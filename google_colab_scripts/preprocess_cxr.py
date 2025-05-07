'''
from google.colab import drive
try:
    print("Attempting to unmount Drive...")
    drive.flush_and_unmount()
    print("Drive unmounted.")
except Exception as e:
    print(f"Error during unmount (might not have been mounted): {e}")

print("Remounting Drive...")
drive.mount('/content/drive', force_remount=True)
print("Drive remounted.")

!git clone https://github.com/USERNAME/REPO_NAME...
import os
%cd REPO_NAME/
print(f"Current directory: {os.getcwd()}")

%cd /content/REPO_NAME/datasets/process_mimic/
!python resize_images.py

%cd /content/REPO_NAME/datasets/process_mimic/
!python create_cxr_splits.py
'''