import os
import shutil
import subprocess
import getpass
import time 
# from google.colab import drive

print("--- Mounting Google Drive ---")
try:
    # drive.mount('/content/drive', force_remount=True) # force_remount can help if already mounted
    print("Google Drive mounted successfully.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    # Stop execution if Drive mounting fails
    raise SystemExit("Halting: Google Drive mount failed.")

# --- Configuration ---
# UPDATE THIS with the correct GCS path you found!
GCS_BASE_PATH = "" # << GOOGLE CLOUD BUCKET URL
# The specific pXX folder subset you want
SUBSET_FOLDER = "p10"
# Target directory IN Google Drive where the 'files' directory structure will be created/used
DRIVE_TARGET_BASE = '/content/drive/[YOUR DATA LOCATION]'
# --- End Configuration ---

# Path for the 'files' directory within your Drive target base
drive_target_files_dir = os.path.join(DRIVE_TARGET_BASE, 'files')
# Full path to where the specific subset folder will be copied in Drive
drive_destination_path = os.path.join(drive_target_files_dir, SUBSET_FOLDER)
# Full path of the source subset folder in GCS
gcs_source_path = f"{GCS_BASE_PATH}/{SUBSET_FOLDER}"

# Create the target directory structure in Drive if it doesn't exist
os.makedirs(drive_destination_path, exist_ok=True)

print(f"GCS Source Path: {gcs_source_path}")
print(f"Google Drive Destination Path: {drive_destination_path}")

# Set environment variables for the bash cell coming next
os.environ['GCS_SOURCE_PATH'] = gcs_source_path
os.environ['DRIVE_DEST_PATH'] = drive_destination_path

# --- Start Timer ---
print(f"\n--- Preparing GCS Transfer for Subset: {SUBSET_FOLDER} ---")
start_time = time.time()

print("--- Authenticating Google Cloud SDK ---")
print("This command will provide a URL.")
print("1. Click the URL.")
print("2. Log in with the Google account that has permission for the GCS bucket.")
print("3. Grant permissions.")
print("4. Copy the authorization code provided.")
print("5. Paste the code back here in the input box below and press Enter.")

# Use flags for better Colab integration (prints URL, avoids opening browser tab)
# !gcloud auth application-default login --quiet --no-launch-browser

# As an alternative if the above has issues, you could try:
# !gcloud auth login --quiet --no-launch-browser

print("\nGoogle Cloud SDK authentication process completed.")

# %%bash

# echo "Starting GCS copy (with Requester Pays)..."
# # These env vars should be set by the preceding Python cell with GCS/Drive paths
# echo "Source: $GCS_SOURCE_PATH"
# echo "Destination: $DRIVE_DEST_PATH"
# echo "Using Quota Project: $QUOTA_PROJECT_ID"

# # Add the -u flag to specify the quota/billing project
# # Using gsutil -m (multithreaded) cp -r (recursive)
# gsutil -u $QUOTA_PROJECT_ID -m cp -r "${GCS_SOURCE_PATH}/" "${DRIVE_DEST_PATH}"

# echo "GCS copy command finished."