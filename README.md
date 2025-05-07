# Multimodal LSTM Paper Replication
**Replication Team:** Pranav Thatte
</br>

**Paper:** MedFuse: Multi-modal fusion with clinical time-series data
and chest X-ray images
</br>

**Original Repository:** https://github.com/nyuad-cai/MedFuse

#### Paper Citation
N. Hayat, K. J. Geras, and F. E. Shamout, 'MedFuse: Multi-modal fusion with clinical time-series data and chest X-ray images,' *Proceedings of Machine Learning Research*, vol. 182, pp. 1--25, 2022. Available: https://arxiv.org/abs/2207.07027

#### MIMIC-IV EHR Data
Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., and Mark, R. (2021). MIMIC-IV (version 1.0). PhysioNet. https://doi.org/10.13026/s6n6-xd98.

#### MIMIC-CXR-JPG Data
Johnson, A., Lungren, M., Peng, Y., Lu, Z., Mark, R., Berkowitz, S., and Horng, S. (2019). MIMIC-CXR-JPG - chest radiographs with structured labels (version 2.0.0). PhysioNet. https://doi.org/10.13026/8360-t248.


## Contents
- [Introduction](./README.md#introduction)
- [Dependencies](./README.md#dependencies)
- [Data](./README.md#data)

## Introduction
This repository details the effort to reproduce the multimodal healthcare paper, MedFuse. MedFuse aims to predict phenotypes assigned to patients and in-hospital mortality within the first 48 hours of being admitted to an ICU. It does so through processing Electronic Health Records (EHR) timeseries data and thorax radiographs (CXR x-ray images) found in the `MIMIC-IV` and `MIMIC-CXR-JPG` datasets. The paper explores 3 main hypotheses:
> 1. Does multimodal data provide a strict advantage over unimodal data in healthcare applications for the tasks of phenotype and in-hospital mortality prediction?
> 2. Given that labels assigned to multimodal data can differ by modality and they are not collected synchronously in healthcare applications, is it a better idea to train encoders for each modality separately and fuse them in later stages of the architecture?
> 3. In healthcare treatment processes, medical staff often collect data sequentially, as needed, as the patient condition evolves. Does it make sense to model modalities as events in a sequence rather than occurring jointly in time?

The paper’s core contribution involves treating multi-modal data as a series of sequential inputs to an LSTM-based module, in contrast with previous work that created a single combined representation of all available modalities. Additionally, they establish a benchmark for multimodal performance on the tasks of phenotype and in-hospital mortality prediction.

## Dependencies

In this section, information required for installing the necessary dependencies is provided.

1. To load all dependencies for the project, first create an environment using `conda create --name [ENV NAME]` and `conda actimate [ENV NAME]`
2. `pip freeze` has generated `requirements.txt` that can be found in the repository
3. Here is a short list of required packages below:
   - matplotlib 3.10.1
   - numpy 2.2.4
   - pandas 2.2.3
   - pillow 11.2.1
   - python 3.13.2
   - pyyaml 6.0.2
   - scikit-learn 1.6.1
   - scipy 1.15.2
   - torch 2.7.0
   - torchvision 0.22.0
   - tqdm 4.67.1

## Data

To acquire the data, please follow the links for each dataset found at the top of the README. From there, you will need to complete a health research course (1 hr) and sign a few forms before getting access to the datasets. I would suggest to create an umbrella folder where the core code lies in one folder and the data is downloaded in another.
</br>

#### MIMIC-IV
This dataset is roughly 7GB. I would suggest to download it in a folder different from any Github repository for data privacy and size considerations. In order to download the full dataset, you can either directly download it from the website or use the following command:
```
wget -r -N -c -np --user [USERNAME] --ask-password https://physionet.org/files/mimiciv/1.0/
```

Afterwords, the steps from the original paper can be followed for preprocessing as they are documented well: https://github.com/nyuad-cai/MedFuse/blob/main/mimic4extract/README.md

</br>
Here is a brief overview of the steps required:

1. Extract data from `.csv` files for each subject and store them into `data/{SUBJECT_ID}`
```
python -m medfuse.datasets.process_mimic.extract_subjects_iv [path to gunzipped csv files] data/root 
```

2. Now, the data will exist in data/root, we can clean the data (missing information etc.) and break data into episode-level info (each ICU stay)
```
python -m medfuse.datasets.process_mimic.validate_events data/root
python -m medfuse.datasets.process_mimic.extract_episodes_from_subject data/root
```

<!-- brew install wget
wget -r -N -c -np --user [USERNAME] --ask-password https://physionet.org/files/mimiciv/1.0/

gunzip ...

python -m medfuse.datasets.process_mimic.extract_subjects_iv [path to gunzipped csv files] data/root 
python -m medfuse.datasets.process_mimic.validate_events data/root
python -m medfuse.datasets.process_mimic.extract_episodes_from_subject data/root -->