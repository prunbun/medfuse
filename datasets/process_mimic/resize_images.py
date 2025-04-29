# import thread # Not needed
import time
from PIL import Image
import glob
from tqdm import tqdm
import os
from multiprocessing.dummy import Pool as ThreadPool # Use threads

print('Starting image resizing script...')

# --- Configuration for Colab/Drive ---
# Base path where your top-level Drive folder is mounted
drive_mount_path = '/content/drive/MyDrive/'
# Specific dataset directory within Drive
dataset_folder = 'datasets2/mimic-cxr-jpg' # Your specified folder
# Sub-folder containing the downloaded p10 images (relative to dataset_folder)
# Assuming structure is MyDrive/datasets2/mimic-cxr-jpg/files/p10/p10XXXXXX/sYYYYYY/*.jpg
input_image_folder_rel = 'files/p10/p10'
# Sub-folder where resized images will be saved (relative to dataset_folder)
# This will be created if it doesn't exist
output_image_folder_rel = 'resized_p10'

# Construct full absolute paths
input_dir = os.path.join(drive_mount_path, dataset_folder, input_image_folder_rel)
output_dir = os.path.join(drive_mount_path, dataset_folder, output_image_folder_rel)
# --- End Configuration ---

print(f"Input image directory (reading from): {input_dir}")
print(f"Output resized directory (writing to): {output_dir}")

# Create the output directory if it doesn't exist
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")
except OSError as e:
    print(f"ERROR: Could not create output directory {output_dir}: {e}")
    raise SystemExit("Halting: Failed to create output directory.")

# Find already resized images in the *output* directory
# Assumes resized images are saved directly into output_dir (not nested)
resized_pattern = os.path.join(output_dir, '*.jpg')
paths_done = glob.glob(resized_pattern)
print(f"Found {len(paths_done)} already resized images in output directory.")

# Find all source images recursively in the *input* directory
source_pattern = os.path.join(input_dir, '**/*.jpg') # Recursive search
paths_all = glob.glob(source_pattern, recursive=True)
print(f"Found {len(paths_all)} total source images in input directory.")

# Create a set of basenames (filenames only) for quick lookup of completed files
done_files_basenames = {os.path.basename(path) for path in paths_done}

# Filter list of source images to find those not yet processed
paths_to_process = [path for path in paths_all if os.path.basename(path) not in done_files_basenames]
print(f"Number of images left to process: {len(paths_to_process)}")

# --- Function to resize a single image ---
def resize_single_image(source_path):
    """Resizes an image and saves it to the output directory."""
    basewidth = 512
    filename = os.path.basename(source_path) # Get only the filename
    output_filepath = os.path.join(output_dir, filename) # Construct full output path

    try:
        with Image.open(source_path) as img:
            # Ensure image is in RGB mode (common for saving as JPG)
            if img.mode != 'RGB':
                 img = img.convert('RGB')

            # Calculate new height maintaining aspect ratio
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))

            # Resize using a high-quality filter
            img_resized = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)

            # Save the resized image
            img_resized.save(output_filepath)
            return filename # Return filename on success for tracking
    except Exception as e:
        # Log error for specific file but continue processing others
        print(f"ERROR processing {source_path}: {e}")
        return None # Indicate failure for this file

# --- Processing Loop ---
threads = 10 # Number of parallel threads (adjust based on Colab performance)
processed_count = 0
error_count = 0

# Check if there are any images left to process
if not paths_to_process:
    print("No new images to process.")
else:
    print(f"\nStarting processing of {len(paths_to_process)} images using {threads} threads...")
    start_time = time.time()

    # Use tqdm for progress bar
    with tqdm(total=len(paths_to_process), desc="Resizing Images") as pbar:
        # Process images in batches using a thread pool
        for i in range(0, len(paths_to_process), threads):
            paths_subset = paths_to_process[i : i + threads]
            pool = ThreadPool(len(paths_subset)) # Pool size for current batch
            results = pool.map(resize_single_image, paths_subset)
            pool.close()
            pool.join()

            # Update counts based on results from the batch
            success_in_batch = sum(1 for r in results if r is not None)
            processed_count += success_in_batch
            error_count += len(paths_subset) - success_in_batch
            pbar.update(len(paths_subset)) # Update progress bar

    end_time = time.time()
    print(f"\nProcessing finished in {end_time - start_time:.2f} seconds.")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors encountered: {error_count}")

print("\nResize script finished.")