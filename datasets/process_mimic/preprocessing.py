import numpy as np
import re
import pandas as pd
from pandas import DataFrame, Series


def add_hcup_ccs_2015_groups(diagnoses, definitions):
    '''
    reads the icd_9_10_definitions_2.yaml file and fetches the definition map and use_in_benchmark
    '''

    def_map = {}
    for dx in definitions:
        for code in definitions[dx]['codes']:
            def_map[code] = (dx, definitions[dx]['use_in_benchmark'])
    
    diagnoses['HCUP_CCS_2015'] = diagnoses.icd_code.apply(lambda c: def_map[c][0] if c in def_map else None)
    diagnoses['USE_IN_BENCHMARK'] = diagnoses.icd_code.apply(lambda c: int(def_map[c][1]) if c in def_map else None)
    
    return diagnoses

def make_phenotype_label_matrix(phenotypes, stays=None):
    '''
    generates a multi-hot matrix where there is an icu_stay vs. phenotypes matrix, where 1 means it is present
    '''

    phenotypes = phenotypes[['stay_id', 'HCUP_CCS_2015']].loc[phenotypes.USE_IN_BENCHMARK > 0].drop_duplicates()
    phenotypes['value'] = 1
    phenotypes = phenotypes.pivot(index='stay_id', columns='HCUP_CCS_2015', values='value')
    if stays is not None:
        phenotypes = phenotypes.reindex(stays.stay_id.sort_values())
    
    return phenotypes.fillna(0).astype(int).sort_index(axis=0).sort_index(axis=1)