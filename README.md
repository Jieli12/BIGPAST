# BIGPAST

This repository **BIGPAST** contains the Python scripts to reproduce the simulation studies in our paper **Bayesian Inference General Procedures for A Single-subject Test Study**. In the following sections, we will frequently reference the document referred to as the **main paper**.

## Requirement

This repository depends on the Python package [skewt-scipy](https://pypi.org/project/skewt-scipy/). To install it, please run the following command in terminal.

```bash
pip install skewt-scipy
```

## Usages

The main paper, **Bayesian Inference General Procedures for A Single-subject Test Study**, includes four numerical studies. This guide will walk you through the process of using Python scripts to reproduce these studies step by step. Please note that these scripts have been tested on a Mac operating system (macOS Sonoma Version 14.4.1).

### Simulation for Jeffery's Prior Comparison Study

This simulation contrasts our implementation of Jeffery's prior with other existing priors. The script `sim_3_1.py` executes the complete simulation as outlined in Section 3.1 of the main paper, storing the results in the `Data` directory. To generate Table 1 from the main paper, please execute the `sim_3_1_result.py` script as follows:

```bash
python sim_3_1.py
python sim_3_1_result.py
```

### Comparison with the existing approaches
