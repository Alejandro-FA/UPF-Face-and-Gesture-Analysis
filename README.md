#  UPF - Face and Gesture Analysis: practices repository

## Installation instructions

### conda

To **replicate the development environment** simply run the following commands (you can change the name of the environment from `face_analysis` to something else):

```bash
conda env create --name face_analysis --file environment.yml
conda config --env --add channels conda-forge
conda config --env --add channels pytorch
conda activate face_analysis
```

### pip

Alternatively, we also provide a `pip` `requirements.txt` file. Please take into account that the project has been developed with `python 3.11`. We have not tested if the code works with other versions of `python`. To **replicate the development environment** simply run the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
