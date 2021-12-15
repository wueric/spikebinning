# spikebinning

This package provides fast functions with a Python interface to bin neural spike trains into specified intervals for model fitting 
(LNPs, GLMs, neural networks, etc). The implementation is in C++ with a bit of parallelism, so it is fast enough to make
very quick work of binning long datasets (more than an hour) at fairly fast bin rates (1 kHz). 
The package also provides tools for merging spike trains of oversplits.

#### Dependencies
1. pybind11
2. numpy

#### Installation

```shell script
python setup.py install --force
```
