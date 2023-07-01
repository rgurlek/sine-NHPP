# Fitting sinusoidal nonhomogeneous Poisson processes (NHPPs) to data
This repository provides Python code for implementing the estimation procedure from [Chen et al. (2023)](#suggested-citations). The tutorial demonstrates how to estimate the arrival rate of customers from a simulated dataset.

Please refer to Appendix A of [Chen et al. (2023)](#suggested-citations) for details of the procedure, which is a simpler variant of the one proposed in [Chen, Lee, and Negahban (2019)](#suggested-citations).


## Suggested citations
- Chen, Gurlek, Lee, Shen (2023): [Can customer arrival rates be modelled by sine waves?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3125120) (Joint issue in *Service Science* and *Stochastic Systems*, forthcoming)

- Chen, Lee, Negahban (2019): [Super-resolution estimation of cyclic arrival rates](https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-3/Super-resolution-estimation-of-cyclic-arrival-rates/10.1214/18-AOS1736.full) (*Annals of Statistics*  47:3:1754-1775)


## Configuring the Python environment

We recommend creating a virtual Python environment to ensure the correct versioning of all dependencies. To this end, run the lines below in an Anaconda Prompt
to create a virtual environment called `sine-NHPP`, and then activate it.
```
conda create -n sine-NHPP python=3.9.13
conda activate sine-NHPP
```


Clone this repository, or manually download the files and extract them to a directory called `sine-NHPP`. Then go to the directory:
```
cd sine-NHPP
```

Install the dependencies by running the following lines in the Anaconda Prompt:
```
pip install numpy==1.21.5
pip install pandas==2.0.2
pip install matplotlib==3.7.1
pip install jupyter
```


Run the tutorial *sine-NHPP_tutorial.ipynb* for a demonstration of the estimation procedure.
```
jupyter notebook sine-NHPP_tutorial.ipynb
```
