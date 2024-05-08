# BIGPAST

This repository **BIGPAST** contains the Python scripts to reproduce the simulation studies in our paper **Bayesian Inference General Procedures for A Single-subject Test Study** [[1]](#1). In the following sections, we will frequently reference the document referred to as the **main paper**.

## Requirement

This repository relies on the Python package [skewt-scipy](https://pypi.org/project/skewt-scipy/). To install this package, execute the following command in your terminal:

```bash
pip install skewt-scipy
```

## Usages

The main paper, **Bayesian Inference General Procedures for A Single-subject Test Study**, includes four numerical studies. This guide will walk you through the process of using Python scripts to reproduce these simulation studies step by step. Please note that we these scripts have tested these scripts on a Mac operating system (macOS Sonoma Version 14.4.1).

### Simulation for Jeffery's Prior Comparison Study

This simulation contrasts our implementation of Jeffery's prior with other existing priors. The script `sim_3_1.py` executes the complete simulation as outlined in Section 3.1 of the main paper, storing the results in the `Data` directory. To generate Table 1 from the main paper, please execute the `sim_3_1_result.py` script as follows:

```bash
python sim_3_1.py
python sim_3_1_result.py
```

### Comparison with the existing approaches

This simulation aims to compare the results of BIGPAST against the results of the $t$-score [[1]](#1), Crawford and Garthwaite Bayesian framework [[2]](#2), and Anderson-Darling non-parametric method [[3]](#3).

To generate Tables 2 and 3 in the main paper, please execute

```bash
python sim_3_2.py
python sim_3_2_mixed_twosided.py
```

Furthermore, one can also generate Tables S2 and S3 by running

```bash
python sim_3_2_mixed_greater.py
python sim_3_2_mixed_less.py
```

### Model misspecification error

Section 3.3 of the main paper studies the model misspecification error if the underlying distribution is skewed Student's $t$ distribution, but one still carries out the $t$ test by assuming the normal distribution.

To fully replicate the results presented in Figure 1 of the main paper, please execute the scripts in the following sequence:

```bash
python sim_3_3.py
python sim_3_3_plot.py
The figures can be found in the `figures` directory.
```

We also provide the intermediate results needed to generate Figure 1 from the main paper. This allows you to swiftly reproduce Figure 1 by simply executing the following command:

```bash
# Please do not run sim_3_3.py as it will rewrite our intermediate results.
python sim_3_3_plot.py
```

The figures can be found in the `figures` directory.

### Comparison study with other frameworks

This section is dedicated to assessing the performance of the proposed BIGPAST methodology and existing approaches when a control group is present. To generate the row results in Table 4 of the main paper, one can run

```bash
python bayes_procedure.py -a 10 -d 10 -n 100 -al two_sided
```

The explanations for these flags are as follows:

- `-a`    the skewness parameter $\alpha$
- `-d`    the degrees of freedom $\nu$
- `-n`    the number of control groups $n$
- `-al`   the direction of the alternative hypothesis: `less`, `greater` or `two-sided`.

Executing the above command will automatically store the results in the `Data` directory. To reproduce Table 4 from the main paper, you can experiment with all the pairs of $(\alpha, \nu)$ listed in Table 4. Please be aware that the default setting for parallel computation utilizes 90 CPU cores. Adjust this setting to match your computer's operating environment. Finally, to generate Figure 3 of the main paper, just run

```bash
python sim_3_4_plot.py
```

Alternatively, if you prefer not to wait for the computations to complete on your machine, we've provided intermediate results in the `Data` directory. In this case, you can simply execute the following command:

```bash
# Please do not run sim_3_4.py as it will rewrite our intermediate results.
python sim_3_4_plot.py
```

The figures can be found in the `figures` directory.

### References

<a id="1">[1]</a>
Li, J., Green, G., Carr, S., Liu, P., & Zhang, J. (2024). Bayesian inference general procedures for a single-subject test study. submitted.

<a id="2">[2]</a>
Crawford, J. R., & Howell, D. C. (1998). Comparing an individual’s test score against norms derived
from small samples. The Clinical Neuropsychologist, 12(4), 482–486. <https://doi.org/10.1076/clin.12.4.482.7241>

<a id="3">[3]</a>
Crawford, J. R., & Garthwaite, P. H. (2007). Comparison of a single case to a control or normative sample in neuropsychology: Development of a Bayesian approach. Cognitive Neuropsychology, 24(4), 343–372. <https://doi.org/10.1080/02643290701290146>

<a id="4">[4]</a>
Scholz, F. W., & Stephens, M. A. (1987). K-sample anderson-darling tests. Journal of the American Statistical Association, 82(399), 918–924. <http://www.jstor.org/stable/2288805>
