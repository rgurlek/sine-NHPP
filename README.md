# sine-NHPP
This GitHub repository provides the code to implement the estimation procedure from the paper
[Can customer arrival rates be modelled by sine waves?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3125120)
(forthcoming joint issue in *Service Science* and *Stochastic Systems*). The code is accompanied by a tutorial to
demonstrate how the rate of customer arrivals to a call center can be estimated using this procedure.

Please refer to Appendix A of the [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3125120) for details of the procedure, which is a simpler
variant of the one proposed in
[Super-resolution estimation of cyclic arrival rates](https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-3/Super-resolution-estimation-of-cyclic-arrival-rates/10.1214/18-AOS1736.full)
(*Annals of Statistics* 2019).

## Configuration

We recommend creating a virtual Python environment to ensure the correct versioning for the dependencies. To this end, run the lines below in an Anaconda Prompt
to create a virtual environment called `sine-NHPP` and to activate it.
```
conda create -n sine-NHPP python=3.9.13
conda activate sine-NHPP
```
Install the dependencies by running the following lines
```
pip install numpy==1.21.5
pip install pandas==2.0.2
Pip install matplotlib==3.7.1
```
After configuring the environment and downloading the files in the GitHub repository, make sure to run the
[tutorial](https://github.com/rgurlek/sine-NHPP/blob/main/sine-NHPP_tutorial.ipynb) within the `sine-NHPP` environment.
