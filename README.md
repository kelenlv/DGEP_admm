# DGEP_admm

This is the Matlab code for the paper [*Consensus-based Distributed Algorithm for Generalized Eigenvalue Problem*](https://doi.org/10.1016/j.sigpro.2023.109307), which is fulfilled by the ADMM framework.

If our work is helpful for your research, please consider citing:

```
@article{LV2024109307,
title = {Consensus-based distributed algorithm for GEP},
journal = {Signal Processing},
volume = {216},
pages = {109307},
year = {2024},
issn = {0165-1684},
doi = {https://doi.org/10.1016/j.sigpro.2023.109307},
url = {https://www.sciencedirect.com/science/article/pii/S016516842300381X},
author = {Kexin Lv and Fan He and Xiaolin Huang and Jie Yang}
}
```


The results of exp1-3 are included in the paper. 

_____

exp1.m: convergence of DGEP+admm in synthetic data

exp2.m: scalability of DGEP+admm in synthetic data

exp3.m: distributed FDA in real-world data

exp3_2.m: distributed FDA in FDA-generated data

FDAproblem_2class.m: generate synthetic FDA data with classes
