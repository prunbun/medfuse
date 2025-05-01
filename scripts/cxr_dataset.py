import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
# import 
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import argparse


# class MIMICCXR(Dataset):
#     def __init__(self, paths, args, transform=None, split='train'):
#         self.data_dir = args.cxr_data_dir
#         self.args = args
#         self.CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
#        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
#        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
#        'Pneumonia', 'Pneumothorax', 'Support Devices']
#         self.filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}

#         metadata = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-metadata.csv')
#         labels = pd.read_csv(f'{self.data_dir}/mimic-cxr-2.0.0-chexpert.csv')
#         labels[self.CLASSES] = labels[self.CLASSES].fillna(0)
#         labels = labels.replace(-1.0, 0.0)
        
#         splits = pd.read_csv(f'{self.data_dir}/cxr_phenotype_split.csv')


#         metadata_with_labels = metadata.merge(labels[self.CLASSES+['study_id'] ], how='inner', on='study_id')


#         self.filesnames_to_labels = dict(zip(metadata_with_labels['dicom_id'].values, metadata_with_labels[self.CLASSES].values))
#         self.filenames_loaded = splits.loc[splits.split==split]['dicom_id'].values
#         self.transform = transform
#         self.filenames_loaded = [filename  for filename in self.filenames_loaded if filename in self.filesnames_to_labels]

#     def __getitem__(self, index):
#         if isinstance(index, str):
#             img = Image.open(self.filenames_to_path[index]).convert('RGB')
#             labels = torch.tensor(self.filesnames_to_labels[index]).float()

#             if self.transform is not None:
#                 img = self.transform(img)
#             return img, labels
        
#         filename = self.filenames_loaded[index]
        
#         img = Image.open(self.filenames_to_path[filename]).convert('RGB')

#         labels = torch.tensor(self.filesnames_to_labels[filename]).float()

#         if self.transform is not None:
#             img = self.transform(img)
#         return img, labels
    
#     def __len__(self):
#         return len(self.filenames_loaded)


# def get_transforms(args):
#     normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     train_transforms = []
#     train_transforms.append(transforms.Resize(256))
#     train_transforms.append(transforms.RandomHorizontalFlip())
#     train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
#     train_transforms.append(transforms.CenterCrop(224))
#     train_transforms.append(transforms.ToTensor())
#     train_transforms.append(normalize)      


#     test_transforms = []
#     test_transforms.append(transforms.Resize(args.resize))


#     test_transforms.append(transforms.CenterCrop(args.crop))

#     test_transforms.append(transforms.ToTensor())
#     test_transforms.append(normalize)


#     return train_transforms, test_transforms

# def get_cxr_datasets(args):
#     train_transforms, test_transforms = get_transforms(args)

#     data_dir = args.cxr_data_dir
    
#     # filepath = f'{args.cxr_data_dir}/paths.npy'
#     # if os.path.exists(filepath):
#     #     paths = np.load(filepath)
#     # else:
#     paths = glob.glob(f'{data_dir}/mimic-cxr-jpg/resized_p10/*.jpg', recursive = True)
#     # np.save(filepath, paths)
    
#     dataset_train = MIMICCXR(paths, args, split='train', transform=transforms.Compose(train_transforms))
#     dataset_validate = MIMICCXR(paths, args, split='validate', transform=transforms.Compose(test_transforms),)
#     dataset_test = MIMICCXR(paths, args, split='test', transform=transforms.Compose(test_transforms),)

#     return dataset_train, dataset_validate, dataset_test

class MIMICCXR(Dataset):
    # Keep the __init__ signature the same
    def __init__(self, available_paths, args, transform=None, split='train'):
        self.args = args
        # meta_data_dir now correctly points to the directory containing the CSVs
        # e.g., /content/drive/MyDrive/datasets2
        self.meta_data_dir = args.cxr_data_dir
        # Construct the image directory path relative to the meta_data_dir
        # Assumes 'mimic-cxr-jpg/resized_p10' is the relative path
        self.image_dir = os.path.join(self.meta_data_dir, 'mimic-cxr-jpg', 'resized_p10')

        self.CLASSES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
                       'Pneumonia', 'Pneumothorax', 'Support Devices']

        print(f"\n[{split} split] Initializing dataset...")
        print(f"[{split} split] Image directory: {self.image_dir}") # Correct image path
        print(f"[{split} split] Metadata/Split/Label directory: {self.meta_data_dir}") # Correct metadata path

        # 1. Build dictionary of AVAILABLE images (filename stem -> full path)
        self.filenames_to_path = {os.path.basename(path).split('.')[0]: path for path in available_paths}
        available_dicom_ids = set(self.filenames_to_path.keys())
        print(f"[{split} split] Found {len(available_dicom_ids)} available image files from paths.")
        # (rest of checks as before)

        # 2. Load Labels (e.g., CheXpert) - use self.meta_data_dir
        # Use getattr to safely get label filename from args 
        label_filename = getattr(args, 'cxr_label_file', 'mimic-cxr-2.0.0-chexpert.csv') 
        label_file_path = os.path.join(self.meta_data_dir, label_filename) # Path is now correct
        print(f"[{split} split] Loading labels from: {label_file_path}")
        try:
            labels = pd.read_csv(label_file_path)
            labels[self.CLASSES] = labels[self.CLASSES].fillna(0)
            labels = labels.replace(-1.0, 0.0)
            labels = labels[['study_id', 'subject_id'] + self.CLASSES]
        except Exception as e:
            print(f"ERROR [{split} split]: Failed to load or process labels file {label_file_path}: {e}")
            raise e

        # 3. Load Metadata - use self.meta_data_dir
        metadata_filename = 'mimic-cxr-2.0.0-metadata.csv'
        metadata_file_path = os.path.join(self.meta_data_dir, metadata_filename) # Path is now correct
        print(f"[{split} split] Loading metadata from: {metadata_file_path}")
        try:
             metadata = pd.read_csv(metadata_file_path)
             metadata = metadata[['dicom_id', 'study_id', 'subject_id']]
        except Exception as e:
             print(f"ERROR [{split} split]: Failed to load metadata file {metadata_file_path}: {e}")
             raise e

        # 4. Merge metadata and labels (logic remains the same)
        print(f"[{split} split] Merging metadata and labels...")
        metadata_with_labels = metadata.merge(labels, how='inner', on=['study_id', 'subject_id'])

        # --- CORRECTED Dictionary Creation ---
        print(f"[{split} split] Creating dicom_id -> labels map...")
        # Get dicom_ids as a NumPy array of strings
        dicom_ids_str = metadata_with_labels['dicom_id'].astype(str).values
        # Get label data as a NumPy array (rows x num_classes)
        label_data_array = metadata_with_labels[self.CLASSES].values
        # Create dictionary by zipping them
        self.filesnames_to_labels = {dicom_id: label_values
                                        for dicom_id, label_values
                                        in zip(dicom_ids_str, label_data_array)}

        # 5. Load the specific SPLIT definition file - use self.meta_data_dir
        # split_filename = args.cxr_split_name # e.g., 'cxr_filtered_relabelled_by_ehr_split.csv'
        split_file_path = os.path.join(args.cxr_data_dir, 'cxr_phenotype_split.csv') # Path is now correct
        print(f"[{split} split] Loading split definition from: {split_file_path}")
        try:
            splits_df = pd.read_csv(split_file_path)
            splits_df['dicom_id'] = splits_df['dicom_id'].astype(str)
            dicom_ids_in_split_file = set(splits_df.loc[splits_df['split'] == split]['dicom_id'].unique())
            print(f"[{split} split] Found {len(dicom_ids_in_split_file)} dicom_ids designated for this split in {split_filename}.")
        except Exception as e:
            print(f"ERROR [{split} split]: Failed to load or process split file {split_file_path}: {e}")
            raise e

        # 6. Final Filter (logic remains the same)
        print(f"[{split} split] Filtering based on intersection of: split file IDs, available image IDs, labeled IDs...")
        final_dicom_ids = list(dicom_ids_in_split_file.intersection(available_dicom_ids).intersection(labeled_dicom_ids))
        self.filenames_loaded = final_dicom_ids

        print(f"[{split} split] Final number of usable samples for this split: {len(self.filenames_loaded)}")
        # (rest of checks as before)

        self.transform = transform

    # __getitem__ and __len__ remain the same as the previous corrected version
    def __getitem__(self, index):
        filename_stem = self.filenames_loaded[index]
        try:
            image_path = self.filenames_to_path[filename_stem]
            labels_array = self.filesnames_to_labels[filename_stem]
            labels = torch.tensor(labels_array).float()
        except KeyError as e:
            print(f"ERROR in __getitem__: KeyError looking up data for dicom_id '{filename_stem}'.")
            raise e
        except Exception as e:
             print(f"ERROR in __getitem__ processing dicom_id '{filename_stem}': {e}")
             raise e
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
             print(f"ERROR in __getitem__: Failed to open or convert image {image_path}: {e}")
             raise e
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'lab': labels}

    def __len__(self):
        return len(self.filenames_loaded)


# --- Helper to define transforms ---
def get_transforms(args):
    """Gets PyTorch transforms for training and validation/test."""
    # Define normalization based on ImageNet stats
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Define training transforms (example, adjust as needed)
    train_transform_list = [
        transforms.Resize(args.resize), # Resize shortest edge to 256? Or Resize(256) for square? Check effect.
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.CenterCrop(args.crop), # Crop to 224x224
        transforms.ToTensor(),
        normalize,
    ]

    # Define validation/test transforms
    test_transform_list = [
        transforms.Resize(args.resize), # Typically resize slightly larger than crop
        transforms.CenterCrop(args.crop), # Center crop to final size
        transforms.ToTensor(),
        normalize,
    ]

    return transforms.Compose(train_transform_list), transforms.Compose(test_transform_list)


# --- Main function to create datasets ---
def get_cxr_datasets(args):
    """Creates the train, validation, and test MIMICCXR datasets."""
    train_transforms, test_transforms = get_transforms(args)

    # args.cxr_data_dir should be the base path, e.g., /content/drive/MyDrive/datasets2
    base_data_dir = args.cxr_data_dir
    # Construct the specific path to the resized images directory
    image_dir = os.path.join(base_data_dir, 'mimic-cxr-jpg', 'resized_p10')

    # Verify image directory exists
    if not os.path.isdir(image_dir):
         print(f"ERROR in get_cxr_datasets: Constructed image directory not found: {image_dir}")
         raise FileNotFoundError(f"Image directory not found: {image_dir}")

    print(f"get_cxr_datasets: Searching for available images in: {image_dir}")
    # Construct glob pattern based on the corrected image_dir
    image_pattern = os.path.join(image_dir, '*.jpg') # Assumes flat structure in resized_p10
    available_image_paths = glob.glob(image_pattern)
    # If nested: image_pattern = os.path.join(image_dir, '**/*.jpg'); available_image_paths = glob.glob(image_pattern, recursive=True)

    print(f"get_cxr_datasets: Found {len(available_image_paths)} image paths via glob using pattern {image_pattern}.")
    if not available_image_paths:
        print(f"WARNING in get_cxr_datasets: No images found in {image_dir}.")
        # Raise error if no images found, as datasets will be empty
        raise FileNotFoundError(f"No JPG images found in {image_dir}")

    # Pass the list of available paths and args to the Dataset constructor
    # MIMICCXR.__init__ will use args.cxr_data_dir for metadata/splits/labels
    print("\nInitializing Train Dataset...")
    dataset_train = MIMICCXR(available_image_paths, args, split='train', transform=train_transforms)

    print("\nInitializing Validation Dataset...")
    dataset_validate = MIMICCXR(available_image_paths, args, split='validate', transform=test_transforms)

    print("\nInitializing Test Dataset...")
    dataset_test = MIMICCXR(available_image_paths, args, split='test', transform=test_transforms)

    return dataset_train, dataset_validate, dataset_test