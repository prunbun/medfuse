import numpy as np
import argparse
import os
import re
import sys

# Get the absolute path of the directory containing this script (scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory (medfuse/) which is the project root
project_root = os.path.dirname(script_dir)

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[fusion_main.py] Added project root to sys.path: {project_root}")

from trainers.fusion_trainer import FusionTrainer
# from trainers.mmtm_trainer import MMTMTrainer
# from trainers.daft_trainer import DAFTTrainer

from scripts.ehr_preprocessing import Discretizer, Normalizer
from scripts.ehr_dataset import get_datasets
from scripts.cxr_dataset import get_cxr_datasets
from scripts.fusion_dataset import load_cxr_ehr
from pathlib import Path
import torch

from arguments import args_parser

# parser = args_parser()
# # add more arguments here ...
# args = parser.parse_args()
# print(args)

# if args.missing_token is not None:
#     from trainers.fusion_tokens_trainer import FusionTokensTrainer as FusionTrainer
    
# path = Path(args.save_dir)
# path.mkdir(parents=True, exist_ok=True)

# seed = 1002
# torch.manual_seed(seed)
# np.random.seed(seed)

# def read_timeseries(args):
#     path = f'{args.ehr_data_dir}/{args.task}/train/10000032_episode1_timeseries.csv'
#     ret = []
#     with open(path, "r") as tsfile:
#         header = tsfile.readline().strip().split(',')
#         assert header[0] == "Hours"
#         for line in tsfile:
#             mas = line.strip().split(',')
#             ret.append(np.array(mas))
#     return np.stack(ret)
    

# discretizer = Discretizer(timestep=float(args.timestep),
#                           store_masks=True,
#                           impute_strategy='previous',
#                           start_time='zero')


# discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
# cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

# normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
# # normalizer_state = args.normalizer_state
# # if normalizer_state is None:
# #     normalizer_state = '../normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
# #     normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
# normalizer_state = args.normalizer_state
# if normalizer_state is None:
#     normalizer_rel_path = os.path.join(os.pardir, 'normalizers', f'ph_ts{args.timestep}.input_str_previous.start_time_zero.normalizer')
#     normalizer_state = os.path.normpath(os.path.join(os.path.dirname(__file__), normalizer_rel_path))
# normalizer.load_params(normalizer_state)

# ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)

# cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)

# train_dl, val_dl, test_dl = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds)

# with open(f"{args.save_dir}/args.txt", 'w') as results_file:
#     for arg in vars(args): 
#         print(f"  {arg:<40}: {getattr(args, arg)}")
#         results_file.write(f"  {arg:<40}: {getattr(args, arg)}\n")

# # if args.fusion_type == 'mmtm':
# #     trainer = MMTMTrainer(
# #         train_dl, 
# #         val_dl, 
# #         args,
# #         test_dl=test_dl
# #         )
# # elif args.fusion_type == 'daft':
# #         trainer = DAFTTrainer(train_dl, 
# #         val_dl, 
# #         args,
# #         test_dl=test_dl)
# # else:
# #     trainer = FusionTrainer(
# #         train_dl, 
# #         val_dl, 
# #         args,
# #         test_dl=test_dl
# #         )
# trainer = FusionTrainer(
#     train_dl, 
#     val_dl, 
#     args,
#     test_dl=test_dl
#     )
# if args.mode == 'train':
#     print("==> training")
#     trainer.train()
# elif args.mode == 'eval':
#     trainer.eval()
# else:
#     raise ValueError("not Implementation for args.mode")
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
print("--- Parsed Arguments ---")
print(args)
print("-----------------------")


# This import depends on the argument, keep it here
if args.missing_token is not None:
    from medfuse.trainers.fusion_tokens_trainer import FusionTokensTrainer as FusionTrainer

path = Path(args.save_dir)
path.mkdir(parents=True, exist_ok=True)

seed = 1002 # Or use args.seed if defined in arguments.py
torch.manual_seed(seed)
np.random.seed(seed)

# Keep the function definition available
def read_timeseries(args):
    # !!! This function still reads a specific file for EHR header generation !!!
    # Make sure this file exists if the EHR block below is executed
    # Or modify this function to return a dummy header if only that is needed
    path = os.path.join(args.ehr_data_dir, args.task, 'train', '10000032_episode1_timeseries.csv') # Example file
    print(f"[fusion_main.py] read_timeseries attempting to read (only if EHR block runs): {path}")
    ret = []
    try:
        with open(path, "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours" # Keep assertion for now
            for line in tsfile:
                mas = line.strip().split(',')
                # Convert to float, handle empty strings
                processed_mas = [float(x) if x else np.nan for x in mas]
                ret.append(np.array(processed_mas))
        if not ret:
             raise ValueError(f"No data read from timeseries file: {path}")
        return np.stack(ret)
    except FileNotFoundError:
        print(f"ERROR in read_timeseries: File not found at {path}. This might be okay if EHR setup is skipped.")
        # Return a dummy structure or raise error depending on how discretizer.transform handles it
        # Returning None might cause issues later, returning dummy header might work if only header needed.
        # Let's raise for now, assuming the dummy file should exist if this path is needed.
        raise
    except Exception as e:
        print(f"ERROR in read_timeseries reading {path}: {e}")
        raise


# --- Initialize EHR variables to None ---
discretizer = None
normalizer = None
ehr_train_ds, ehr_val_ds, ehr_test_ds = None, None, None
# --- End Initialization ---


# --- Conditional EHR Setup START ---
# Check if EHR data processing is actually needed based on arguments
needs_ehr = args.data_pairs != 'radiology' and args.fusion_type != 'uni_cxr'
# Add other conditions if needed, e.g. and args.fusion_type != 'some_other_cxr_only_mode'

if True:
    print("\n[fusion_main.py] Setting up EHR discretizer/normalizer...")
    try:
        discretizer = Discretizer(timestep=float(args.timestep),
                                  store_masks=True,
                                  impute_strategy=args.imputation, # Use arg
                                  start_time='zero')
                                  # Assuming Discretizer default config path is fixed or file exists

        # Note: read_timeseries will be called here by discretizer.transform
        discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)
        normalizer_state = args.normalizer_state
        if normalizer_state is None:
            # Construct default path relative to this script's dir's PARENT (project root)
            norm_filename = f'ph_ts{args.timestep}.input_str_{args.imputation}.start_time_zero.normalizer'
            # Assumes 'normalizers' is directly under project_root (/content/medfuse/normalizers/)
            normalizer_state = os.path.join(project_root, 'normalizers', norm_filename)
            print(f"[fusion_main.py] Using default normalizer path: {normalizer_state}")
        else:
            # Assume normalizer_state is a full path if provided
             print(f"[fusion_main.py] Using provided normalizer path: {normalizer_state}")

        # Ensure the normalizer file actually exists before loading
        if not os.path.exists(normalizer_state):
             raise FileNotFoundError(f"Normalizer state file not found: {normalizer_state}")
        normalizer.load_params(normalizer_state)

        # Load EHR datasets only if setup was successful
        ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)
        print("[fusion_main.py] Finished EHR setup.")

    except Exception as e:
        print(f"\nERROR during EHR setup block: {type(e).__name__} - {e}")
        print("This might be okay if only CXR is being used, but check paths/files if EHR is needed.")
        # Decide whether to halt or continue if EHR setup fails but isn't strictly needed
        # For now, let's allow continuing, but CXR loading might fail if it depends on e.g. args.num_classes
        # sys.exit(1) # Uncomment this to halt on any EHR setup error
else:
    print("\n[fusion_main.py] Skipping EHR setup based on arguments (e.g., radiology/uni_cxr mode).")
# --- Conditional EHR Setup END ---


# --- CXR Dataset Loading (Runs Regardless) ---
print("\n[fusion_main.py] Loading CXR datasets...")
try:
    # This function needs to correctly use args like cxr_data_dir, cxr_split_name, etc.
    cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)
    print("[fusion_main.py] Finished loading CXR datasets.")
except Exception as e:
    print(f"FATAL ERROR loading CXR datasets: {type(e).__name__} - {e}")
    print("Check paths in arguments (cxr_data_dir) and implementation of get_cxr_datasets.")
    sys.exit(1)


# --- Dataloader Creation ---
print("\n[fusion_main.py] Creating dataloaders...")
try:
    # This function MUST handle ehr_*_ds variables being None if EHR setup was skipped
    train_dl, val_dl, test_dl = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds)
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}, Test batches: {len(test_dl)}")
except Exception as e:
    print(f"FATAL ERROR creating dataloaders: {type(e).__name__} - {e}")
    print("Check implementation of load_cxr_ehr, especially handling of None datasets.")
    sys.exit(1)


# --- Save Arguments ---
# Moved argument saving here, after potential errors during data loading
print("\n[fusion_main.py] Saving arguments...")
try:
    with open(os.path.join(args.save_dir, "args.txt"), 'w') as results_file:
        for arg in vars(args):
            # Print required for Colab output, write required for saving
            print(f"  {arg:<40}: {getattr(args, arg)}")
            results_file.write(f"  {arg:<40}: {getattr(args, arg)}\n")
except Exception as e:
     print(f"Warning: Could not save arguments to {args.save_dir}/args.txt: {e}")
# --- End Save Arguments ---


# --- Trainer Initialization ---
print("\n[fusion_main.py] Initializing trainer...")
try:
    # Removed MMTM/DAFT options based on user's previous script version
    # This assumes FusionTrainer handles the 'uni_cxr' logic based on args
    trainer = FusionTrainer(
        train_dl,
        val_dl,
        args,
        test_dl=test_dl
        )
    print(f"[fusion_main.py] Trainer initialized: {type(trainer).__name__}")
except Exception as e:
     print(f"FATAL ERROR initializing trainer: {type(e).__name__} - {e}")
     print("Check trainer implementation and model compatibility.")
     sys.exit(1)


# --- Run Mode ---
print(f"\n[fusion_main.py] Running mode: {args.mode}")
try:
    if args.mode == 'train':
        print("==> training")
        trainer.train()
    elif args.mode == 'eval':
        print("==> evaluating")
        trainer.eval()
    else:
        # Argparse choices should prevent this, but added for safety
        raise ValueError(f"Invalid mode specified: {args.mode}")
except Exception as e:
     print(f"FATAL ERROR during {args.mode}: {type(e).__name__} - {e}")
     # Consider printing traceback for detailed debugging
     # import traceback
     # traceback.print_exc()
     sys.exit(1)

print("\n--- fusion_main.py Finished ---")