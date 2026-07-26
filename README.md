# Overview
This repository contains code for stochastic diffusion projects inspired from [De Bruyne et al., 2021](https://iopscience.iop.org/article/10.1088/1742-5468/ac2906).

A Python virtual environment is advised and is assumed in all instructions in this README. Details of required Python versions can be found at the end of this README. This was developed on an Apple Silicon, M1, 2020 and has not been tested on other machines.

For any questions or issues please contact the owner of this repository.

## Getting Started

### Recreating De Bruyne et al., 2021 figures
- Navigate to `code`
- Run `$ python debruyne_fig_#.py` where # is a placeholder for corresponding figure number.
- Output will be open and is saved in `figures`.


## Repository Structure
- Python code exists in the directory `code`
- Figures that are generated from `code` are created in `figures` (make sure this exists).
- Saved data from simulations is stored in `data`

## Python Virtual Environmnet

Python version 3.14.6 was used to implement this repository.

Please see https://docs.python.org/3/library/venv.html for instructions on how to create a virtual environment on your machine.

Then, read in required Python modules via `$ pip install -r requirements.txt`


