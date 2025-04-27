# medfuse

conda create --name medfuse_env
conda activate medfuse_env
pip3 freeze > requirements.txt

brew install wget
wget -r -N -c -np --user [USERNAME] --ask-password https://physionet.org/files/mimiciv/1.0/

gunzip ...

python -m medfuse.datasets.process_mimic.extract_subjects_iv [path to gunzipped csv files] data/root 
python -m medfuse.datasets.process_mimic.validate_events data/root
python -m medfuse.datasets.process_mimic.extract_episodes_from_subject data/root