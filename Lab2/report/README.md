# Compiling the TEX File using latexmk

To compile the TEX file under the `report` folder of the project, you can use `latexmk`. `latexmk` is a command-line tool that automates the process of compiling LaTeX documents.

Follow the steps below to compile the TEX file:

1. Open a terminal or command prompt.
2. Navigate to the `report` folder of the project using the `cd` command:
    ```bash
    cd report
    ```
3. Run the following command to compile the TEX file using `latexmk`:
    ```bash
    latexmk -pdf main.tex
    ```
    This command will compile the `main.tex` file and generate a PDF output.

> **NOTE**: `latexmkrc` is a configuration file used by `latexmk`. We have downloaded the one [provided by Overleaf](https://www.overleaf.com/learn/how-to/How_does_Overleaf_compile_my_project%3F) to have a consistent output. 

## Cleaning temporary files

To clean temporary files generated during the compilation process, you can use the `latexmk` command with the `-c` option.

```bash
latexmk -c
```
This command will remove all the temporary files generated during the compilation process.


## Installation requirements

- For **Linux OSes**: To use `latexmk` on Linux, you need to have two main dependencies installed: `texlive` and `latexmk`. `texlive` is a comprehensive distribution of LaTeX, which includes all the necessary packages and tools for compiling LaTeX documents. `latexmk` is a command-line tool that automates the compilation process.

    To install `texlive` and `latexmk` on Linux, you can use the package manager specific to your Linux distribution. For example, on Ubuntu or Debian-based systems, you can use the following command:

    ```bash
    sudo apt install texlive latexmk
    ```

    This command will install `texlive` and `latexmk` along with their dependencies.

- **For macOS**: On macOS, you need to have MacTeX installed, which is a distribution of LaTeX specifically designed for macOS. MacTeX includes all the necessary packages and tools, including `latexmk`, for compiling LaTeX documents.

    To install MacTeX on macOS, you can download the MacTeX distribution from the official website (https://www.tug.org/mactex/) and follow the installation instructions provided. Once MacTeX is installed, you will have `latexmk` available for compiling your TEX files, so you don't need to install it separately.

With the required dependencies installed, you can follow the steps mentioned in the README to compile your TEX file using `latexmk`.