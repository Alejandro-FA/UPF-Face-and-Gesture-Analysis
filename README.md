#  UPF - Face and Gesture Analysis: practices repository

## Installation instructions

### conda

To **replicate the development environment** simply run the following commands (you can change the name of the environment from `face_analysis` to something else):

```bash
conda env create --name face_analysis --file environment.yml
conda activate face_analysis
conda config --env --add channels conda-forge
conda config --env --add channels pytorch
conda config --env --add channels nvidia # Only needed if you have an Nvidia GPU
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


## Compiling the TEX reports using latexmk

To compile the TEX file under the `report` folder of the project, you can use `latexmk`. `latexmk` is a command-line tool that automates the process of compiling LaTeX documents.

Follow the steps below to compile the TEX file:

1. Open a terminal or command prompt.
2. Navigate to the `report` folder of the project using the `cd` command:
    ```bash
    cd Lab2/report
    ```
3. Run the following command to compile the TEX file using `latexmk`:
    ```bash
    latexmk -pdf main.tex
    ```
    This command will compile the `main.tex` file and generate a PDF output.

> **NOTE**: `latexmkrc` is a configuration file used by `latexmk`. We have downloaded the one [provided by Overleaf](https://www.overleaf.com/learn/how-to/How_does_Overleaf_compile_my_project%3F) to have a consistent output. 

### Having a live preview of the pdf file

If you want Latexmk to continuously check all input files for changes and re-compile the whole thing if needed and always display the result, type

```bash
latexmk -pdf -pvc main.tex
```

Then, whenever you change something in any of your source files and save your changes, the preview is automagically updated. But: This doesn't work with all viewers, especially not with *Adobe Reader*. For more information about this feature check the [official documentation](https://mg.readthedocs.io/latexmk.html#running-latexmk).

### Cleaning temporary files

To clean temporary files generated during the compilation process, you can use the `latexmk` command with the `-c` option.

```bash
latexmk -c
```
This command will remove all the temporary files generated during the compilation process.


### Installation requirements

- For **Linux OSes**: To use `latexmk` on Linux, you need to have two main dependencies installed: `texlive` and `latexmk`. `texlive` is a comprehensive distribution of LaTeX, which includes all the necessary packages and tools for compiling LaTeX documents. `latexmk` is a command-line tool that automates the compilation process.

    To install `texlive` and `latexmk` on Linux, you can use the package manager specific to your Linux distribution. For example, on Ubuntu or Debian-based systems, you can use the following command:

    ```bash
    sudo apt install texlive latexmk
    ```

    This command will install `texlive` and `latexmk` along with their dependencies.

- **For macOS**: On macOS, you need to have MacTeX installed, which is a distribution of LaTeX specifically designed for macOS. MacTeX includes all the necessary packages and tools, including `latexmk`, for compiling LaTeX documents.

    To install MacTeX on macOS, you can download the MacTeX distribution from the official website (https://www.tug.org/mactex/) and follow the installation instructions provided. Once MacTeX is installed, you will have `latexmk` available for compiling your TEX files, so you don't need to install it separately.

With the required dependencies installed, you can follow the steps mentioned in the README to compile your TEX file using `latexmk`.