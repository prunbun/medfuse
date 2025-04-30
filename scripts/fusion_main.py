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

parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
print(args)

if args.missing_token is not None:
    from trainers.fusion_tokens_trainer import FusionTokensTrainer as FusionTrainer
    
path = Path(args.save_dir)
path.mkdir(parents=True, exist_ok=True)

seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)

def read_timeseries(args):
    path = f'{args.ehr_data_dir}/{args.task}/train/10000032_episode1_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)
    

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')


discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
# normalizer_state = args.normalizer_state
# if normalizer_state is None:
#     normalizer_state = '../normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
#     normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_rel_path = os.path.join(os.pardir, 'normalizers', f'ph_ts{args.timestep}.input_str_previous.start_time_zero.normalizer')
    normalizer_state = os.path.normpath(os.path.join(os.path.dirname(__file__), normalizer_rel_path))
normalizer.load_params(normalizer_state)

ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)

cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)

train_dl, val_dl, test_dl = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, cxr_val_ds, ehr_test_ds, cxr_test_ds)

with open(f"{args.save_dir}/args.txt", 'w') as results_file:
    for arg in vars(args): 
        print(f"  {arg:<40}: {getattr(args, arg)}")
        results_file.write(f"  {arg:<40}: {getattr(args, arg)}\n")

# if args.fusion_type == 'mmtm':
#     trainer = MMTMTrainer(
#         train_dl, 
#         val_dl, 
#         args,
#         test_dl=test_dl
#         )
# elif args.fusion_type == 'daft':
#         trainer = DAFTTrainer(train_dl, 
#         val_dl, 
#         args,
#         test_dl=test_dl)
# else:
#     trainer = FusionTrainer(
#         train_dl, 
#         val_dl, 
#         args,
#         test_dl=test_dl
#         )
trainer = FusionTrainer(
    train_dl, 
    val_dl, 
    args,
    test_dl=test_dl
    )
if args.mode == 'train':
    print("==> training")
    trainer.train()
elif args.mode == 'eval':
    trainer.eval()
else:
    raise ValueError("not Implementation for args.mode")