# medfuse

conda create --name medfuse_env
conda activate medfuse_env
pip3 freeze > requirements.txt

brew install wget
wget -r -N -c -np --user [USERNAME] --ask-password https://physionet.org/files/mimiciv/1.0/

gunzip ...