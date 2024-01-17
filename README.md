#  UPF - Face and Gesture Analysis: practices repository

## Installation instructions

### conda

To **replicate the development environment** simply run the following commands (you can change the name of the environment from `face_analysis` to something else):

```bash
conda env create --name face_analysis --file environment.yml
conda activate face_analysis
```

### pip

Alternatively, we also provide a `requirements.txt` file that can be used with `pip`. Please take into account that the project has been developed with `python 3.11`. We have not tested if the code works with other versions of `python`. To **replicate the development environment** simply run the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


| Lab                   | Week | Weight                   |
| --------------------- | ---- | ------------------------ |
| 1-2                   | 1-2  | 15%                      |
| 3                     | 3-4  | 15%                      |
| 4                     | 5-6  | 15%                      |
| Competitive challenge | 7-9  | 25% (grade > 5 required) |
