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
import random

R_CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax', 'Support Devices']

CLASSES = [
       'Acute and unspecified renal failure', 'Acute cerebrovascular disease',
       'Acute myocardial infarction', 'Cardiac dysrhythmias',
       'Chronic kidney disease',
       'Chronic obstructive pulmonary disease and bronchiectasis',
       'Complications of surgical procedures or medical care',
       'Conduction disorders', 'Congestive heart failure; nonhypertensive',
       'Coronary atherosclerosis and other heart disease',
       'Diabetes mellitus with complications',
       'Diabetes mellitus without complication',
       'Disorders of lipid metabolism', 'Essential hypertension',
       'Fluid and electrolyte disorders', 'Gastrointestinal hemorrhage',
       'Hypertension with complications and secondary hypertension',
       'Other liver diseases', 'Other lower respiratory disease',
       'Other upper respiratory disease',
       'Pleurisy; pneumothorax; pulmonary collapse',
       'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
       'Respiratory failure; insufficiency; arrest (adult)',
       'Septicemia (except in labor)', 'Shock'
    ]

class MIMIC_CXR_EHR(Dataset):
    def __init__(self, args, metadata_with_labels, ehr_ds, cxr_ds, split='train'):
        print(f"--- [MIMIC_CXR_EHR Init - {split}] Starting ---")
        self.args = args
        self.split = split
        self.ehr_ds = ehr_ds
        self.cxr_ds = cxr_ds # This instance contains only your available p10 images

        self.CLASSES = CLASSES
        if 'radiology' in args.labels_set:
            self.CLASSES = R_CLASSES

        # 1. Get the set of actually available CXR dicom_ids from your cxr_ds
        # Assumes cxr_ds has the 'filenames_loaded' attribute like your MIMICCXR class
        try:
            available_cxr_ids = set(self.cxr_ds.filenames_loaded)
            print(f"--- [MIMIC_CXR_EHR Init - {split}] Found {len(available_cxr_ids)} available CXR IDs in provided cxr_ds ---")
            if not available_cxr_ids:
                 print(f"--- [MIMIC_CXR_EHR Init - {split}] WARNING: No available CXR IDs found in cxr_ds. Paired modes will likely be empty. ---")

        except AttributeError:
             print(f"--- [MIMIC_CXR_EHR Init - {split}] ERROR: The provided cxr_ds object does not have a 'filenames_loaded' attribute. Cannot determine available CXRs. ---")
             # Handle this error appropriately - perhaps raise it, or default to assuming no CXRs are available
             available_cxr_ids = set() # Or raise Exception(...)


        # 2. Filter the input metadata_with_labels based on available CXR IDs
        print(f"--- [MIMIC_CXR_EHR Init - {split}] Original metadata_with_labels size: {len(metadata_with_labels)} ---")
        # Ensure 'dicom_id' column is string type for consistent matching with filenames_loaded
        metadata_with_labels['dicom_id'] = metadata_with_labels['dicom_id'].astype(str)
        
        metadata_filtered = metadata_with_labels[
            metadata_with_labels['dicom_id'].isin(available_cxr_ids)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning if you modify later
        
        print(f"--- [MIMIC_CXR_EHR Init - {split}] Filtered metadata size (pairs with available CXR): {len(metadata_filtered)} ---")
        
        # Store the filtered metadata if needed elsewhere, or just use it to create paired lists
        self.metadata_with_labels_filtered = metadata_filtered 

        # 3. Create paired lists FROM THE FILTERED metadata
        self.cxr_files_paired = self.metadata_with_labels_filtered['dicom_id'].values
        self.ehr_files_paired = self.metadata_with_labels_filtered['stay'].values
        print(f"--- [MIMIC_CXR_EHR Init - {split}] Number of EHR-CXR pairs where CXR is available: {len(self.cxr_files_paired)} ---")


        # 4. Define all EHR files and unpaired EHR files (based on the *filtered* pairs)
        self.ehr_files_all = self.ehr_ds.names # All EHR files available in ehr_ds
        # Unpaired = All EHR - EHRs that were successfully paired with an AVAILABLE CXR
        self.ehr_files_unpaired = list(set(self.ehr_files_all) - set(self.ehr_files_paired))
        print(f"--- [MIMIC_CXR_EHR Init - {split}] Total EHR files: {len(self.ehr_files_all)} ---")
        print(f"--- [MIMIC_CXR_EHR Init - {split}] Unpaired EHR files (relative to available CXR pairs): {len(self.ehr_files_unpaired)} ---")

        # 5. Set data_ratio based on split (logic remains the same)
        self.data_ratio = self.args.data_ratio
        if split=='test':
            self.data_ratio =  1.0
            print(f"--- [MIMIC_CXR_EHR Init - {split}] Data ratio set to 1.0 for test split ---")
        elif split == 'val':
            self.data_ratio =  0.0
            print(f"--- [MIMIC_CXR_EHR Init - {split}] Data ratio set to 0.0 for val split ---")
        else:
             print(f"--- [MIMIC_CXR_EHR Init - {split}] Data ratio for train split: {self.data_ratio} ---")
        
        print(f"--- [MIMIC_CXR_EHR Init - {split}] Initialization Complete ---")


    def __getitem__(self, index):
        if self.args.data_pairs == 'paired_ehr_cxr':
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
            cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_paired[index]]
            return ehr_data, cxr_data, labels_ehr, labels_cxr
        elif self.args.data_pairs == 'paired_ehr':
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
            cxr_data, labels_cxr = None, None
            return ehr_data, cxr_data, labels_ehr, labels_cxr
        elif self.args.data_pairs == 'radiology':
            ehr_data, labels_ehr = np.zeros((1, 10)), np.zeros(self.args.num_classes)
            cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_all[index]]
            return ehr_data, cxr_data, labels_ehr, labels_cxr
        elif self.args.data_pairs == 'partial_ehr':
            ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_all[index]]
            cxr_data, labels_cxr = None, None
            return ehr_data, cxr_data, labels_ehr, labels_cxr
        
        elif self.args.data_pairs == 'partial_ehr_cxr':
            if index < len(self.ehr_files_paired):
                ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_paired[index]]
                cxr_data, labels_cxr = self.cxr_ds[self.cxr_files_paired[index]]
            else:
                index = random.randint(0, len(self.ehr_files_unpaired)-1) 
                ehr_data, labels_ehr = self.ehr_ds[self.ehr_files_unpaired[index]]
                cxr_data, labels_cxr = None, None
            return ehr_data, cxr_data, labels_ehr, labels_cxr

        
    
    def __len__(self):
        if 'paired' in self.args.data_pairs:
            return len(self.ehr_files_paired)
        elif self.args.data_pairs == 'partial_ehr':
            return len(self.ehr_files_all)
        elif self.args.data_pairs == 'radiology':
            return len(self.cxr_files_all)
        elif self.args.data_pairs == 'partial_ehr_cxr':
            return len(self.ehr_files_paired) + int(self.data_ratio * len(self.ehr_files_unpaired)) 
        


def loadmetadata(args):

    data_dir = args.cxr_data_dir
    cxr_metadata = pd.read_csv(f'{data_dir}/mimic-cxr-2.0.0-metadata.csv')
    icu_stay_metadata = pd.read_csv(f'{args.ehr_data_dir}/root/all_stays.csv')
    columns = ['subject_id', 'stay_id', 'intime', 'outtime']
    
    # only common subjects with both icu stay and an xray
    cxr_merged_icustays = cxr_metadata.merge(icu_stay_metadata[columns ], how='inner', on='subject_id')
    
    # combine study date time
    cxr_merged_icustays['StudyTime'] = cxr_merged_icustays['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
    cxr_merged_icustays['StudyDateTime'] = pd.to_datetime(cxr_merged_icustays['StudyDate'].astype(str) + ' ' + cxr_merged_icustays['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")
    
    cxr_merged_icustays.intime=pd.to_datetime(cxr_merged_icustays.intime)
    cxr_merged_icustays.outtime=pd.to_datetime(cxr_merged_icustays.outtime)
    end_time = cxr_merged_icustays.outtime
    if args.task == 'in-hospital-mortality':
        end_time = cxr_merged_icustays.intime + pd.DateOffset(hours=48)

    cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&((cxr_merged_icustays.StudyDateTime<=end_time))]

    # cxr_merged_icustays_during = cxr_merged_icustays.loc[(cxr_merged_icustays.StudyDateTime>=cxr_merged_icustays.intime)&((cxr_merged_icustays.StudyDateTime<=cxr_merged_icustays.outtime))]
    # select cxrs with the ViewPosition == 'AP
    cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'AP']

    groups = cxr_merged_icustays_AP.groupby('stay_id')

    groups_selected = []
    for group in groups:
        # select the latest cxr for the icu stay
        selected = group[1].sort_values('StudyDateTime').tail(1).reset_index()
        groups_selected.append(selected)
    groups = pd.concat(groups_selected, ignore_index=True)
    # import pdb; pdb.set_trace()

    # groups['cxr_length'] = (groups['StudyDateTime'] - groups['intime']).astype('timedelta64[h]')
    return groups

# def 
def load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds):

    cxr_merged_icustays = loadmetadata(args) 

    # cxr_merged_icustays['cxr_length'] = (cxr_merged_icustays['StudyDateTime'] - cxr_merged_icustays['intime'] ).astype('timedelta64[h]')

    # import pdb; pdb.set_trace()

    splits_labels_train = pd.read_csv(f'{args.ehr_data_dir}/{args.task}/train_listfile.csv')
    splits_labels_val = pd.read_csv(f'{args.ehr_data_dir}/{args.task}/val_listfile.csv')
    splits_labels_test = pd.read_csv(f'{args.ehr_data_dir}/{args.task}/test_listfile.csv')


    train_meta_with_labels = cxr_merged_icustays.merge(splits_labels_train, how='inner', on='stay_id')
    val_meta_with_labels = cxr_merged_icustays.merge(splits_labels_val, how='inner', on='stay_id')
    test_meta_with_labels = cxr_merged_icustays.merge(splits_labels_test, how='inner', on='stay_id')
    
    train_ds = MIMIC_CXR_EHR(args, train_meta_with_labels, ehr_train_ds, cxr_train_ds)
    val_ds = MIMIC_CXR_EHR(args, val_meta_with_labels, ehr_val_ds, cxr_val_ds, split='val')
    test_ds = MIMIC_CXR_EHR(args, test_meta_with_labels, ehr_test_ds, cxr_test_ds, split='test')


    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)
    test_dl = DataLoader(test_ds, args.batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)

    return train_dl, val_dl, test_dl

def printPrevalence(merged_file, args):
    if args.labels_set == 'pheno':
        total_rows = len(merged_file)
        print(merged_file[CLASSES].sum()/total_rows)
    else:
        total_rows = len(merged_file)
        print(merged_file['y_true'].value_counts())
    # import pdb; pdb.set_trace()

def my_collate(batch):
    x = [item[0] for item in batch]
    pairs = [False if item[1] is None else True for item in batch]
    img = torch.stack([torch.zeros(3, 224, 224) if item[1] is None else item[1] for item in batch])
    x, seq_length = pad_zeros(x)
    targets_ehr = np.array([item[2] for item in batch])
    targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
    return [x, img, targets_ehr, targets_cxr, seq_length, pairs]

def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret), seq_length