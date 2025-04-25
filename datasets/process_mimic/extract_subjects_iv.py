'''
FILE DESCRIPTION

This file essentially creates the directory structure data/root/ where data is organized by SUBJECT_ID. 
Inside each subject's folder, it creates initial stays.csv, diagnoses.csv, events.csv
'''

from .mimic3_csv_utils import *
from .preprocessing import *

import argparse
import os
import yaml #pyyaml

'''
PARSE ARGS
'''
parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                    default=['OUTPUTEVENTS', 'CHARTEVENTS', 'LABEVENTS'])
parser.add_argument('--phenotype_definitions', '-p', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/icd_9_10_definitions_2.yaml'),
                    help='YAML file with phenotype definitions.')
parser.add_argument('--itemids_file', '-i', type=str,default=os.path.join(os.path.dirname(__file__),'../resources/itemid_to_variable_map.csv'),
                    help='CSV containing list of ITEMIDs to keep.')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output')
parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suspend printing of details')
parser.set_defaults(verbose=True)
parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1000 subjects, 1000000 events.')
args, _ = parser.parse_known_args()

# ----------------------------------

'''
PREP OUTPUT
'''
try:
    os.makedirs(args.output_path)
except:
    pass

# ----------------------------------

'''
FETCH RAW DATA
'''
patients = read_patients_table(f'{args.mimic3_path}/core/patients.csv')
admits = read_admissions_table(f'{args.mimic3_path}/core/admissions.csv')
stays = read_icustays_table(f'{args.mimic3_path}/icu/icustays.csv')

if args.verbose:
    print('START:\n\tstay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
          stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))
    
# ----------------------------------

'''
CLEAN STAYS DATA BY ADMISSIONS
'''
stays = remove_icustays_with_transfers(stays)

if args.verbose:
    print('REMOVE ICU TRANSFERS:\n\tstay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
          stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))

stays = merge_on_subject_admission(stays, admits)
stays = merge_on_subject(stays, patients)
stays = filter_admissions_on_nb_icustays(stays)

if args.verbose:
    print('REMOVE MULTIPLE STAYS PER ADMIT:\n\tstay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
          stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))


'''
CLEAN STAYS DATA BY PATIENT INFO AND ASSIGN LABELS
'''
stays = add_age_to_icustays(stays)
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
stays = filter_icustays_on_age(stays)

if args.verbose:
    print('REMOVE PATIENTS AGE < 18:\n\tstay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
          stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))
    
'''
WRITE CLEANED STAYS TO .csv
'''
stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)

# ----------------------------------

'''
CLEAN DIAGNOSES DATA
'''

diagnoses = read_icd_diagnoses_table(f'{args.mimic3_path}/hosp')
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)

'''
WRITE CLEANED DIAGNOSES TO .csv
'''
diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)
count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))

# ----------------------------------

'''
PROCESS MAPPINGS BETWEEN CODES AND PHENOTYPES
'''
# Construct the file path
phenotype_defs_path = f'{args.phenotype_definitions}'

# Open the file using 'with' and load YAML safely
try:
    with open(phenotype_defs_path, 'r') as def_file:
        phenotype_definitions = yaml.load(def_file, Loader=yaml.SafeLoader) # Specify the Loader
except FileNotFoundError:
    print(f"Error: Phenotype definitions file not found at {phenotype_defs_path}")
    # Handle the error appropriately, e.g., exit or raise
    raise
except yaml.YAMLError as e:
    print(f"Error parsing YAML file {phenotype_defs_path}: {e}")
    # Handle the error appropriately
    raise

phenotypes = add_hcup_ccs_2015_groups(diagnoses, phenotype_definitions)

make_phenotype_label_matrix(phenotypes, stays).to_csv(os.path.join(args.output_path, 'phenotype_labels.csv'),
                                                      index=False, quoting=csv.QUOTE_NONNUMERIC)

'''
SEPARATE DATA FOR EACH PATIENT
'''
if args.test:
    pat_idx = np.random.choice(patients.shape[0], size=1000)
    patients = patients.iloc[pat_idx]
    stays = stays.merge(patients[['subject_id']], left_on='subject_id', right_on='subject_id')
    args.event_tables = [args.event_tables[0]]
    print('Using only', stays.shape[0], 'stays and only', args.event_tables[0], 'table')

subjects = stays.subject_id.unique()
break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
break_up_diagnoses_by_subject(phenotypes, args.output_path, subjects=subjects)
items_to_keep = set(
    [int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['ITEMID'].unique()]) if args.itemids_file else None

# ----------------------------------
'''
CREATE EVENTS CSV FILES FOR EACH PATIENT
'''
for table in args.event_tables:
    read_events_table_and_break_up_by_subject(f'{args.mimic3_path}', table, args.output_path, items_to_keep=items_to_keep,
                                              subjects_to_keep=subjects)