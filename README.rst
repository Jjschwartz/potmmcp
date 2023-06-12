POTMMCP
#######

This repository contains the implementation of the **Partially Observable Type-based Meta Monte-Carlo Planning (POTMMCP)** algorithm 
from the paper: `Combining a Meta-Policy and Monte-Carlo Planning for Scalable Type-Based Reasoning in Partially Observable Environments <https://arxiv.org/abs/2306.06067>`_.


Installation
------------

The library is implemented in ``python 3.8`` and has the following main dependencies:

1. `Pytorch <https://pytorch.org>`_ >=1.11
2. `Rllib <https://github.com/ray-project/ray/tree/1.12.0>`_ == 1.12
3. `posggym <https://github.com/RDLLab/posggym>`_ (comes with repo)
4. `posggym-agents <https://github.com/Jjschwartz/posggym-agents>`_  (comes with repo)

As with any python package we recommend using a virtual environment (e.g. `Conda <https://docs.conda.io/en/latest/>`_).

**Installation** of ``potmmcp`` requires cloning the repo then installing using pip:

.. code-block:: bash

    git clone git@github.com:Jjschwartz/potmmcp.git

    # install posggym
    cd potmmcp/posggym
    pip install -e .
    
    # install posggym-agents
    cd ../posggym-agents
    pip install -e .

    # install potmmcp
    cd ..
    pip install -e .


This will install the ``potmmcp`` package along with the necessary dependencies.


The codebase
------------

There are two main parts to the codebase:

1. The ``potmmcp`` directory containing the ``potmmcp`` python package
2. The ``experiments`` directory containing scripts and Jupyter notebooks for running and analysing the experiments used in the paper. The results are also store here.


potmmcp
```````

The ``potmmcp`` python package contains a few main parts:

1. ``baselines`` - implementation code for the different baselines used in the paper
2. ``plot`` - code used for generating plots and running analysis
3. ``run`` - code for running and tracking experiments
4. ``tree`` - the implementation of the **POTMMCP** algorithm
5. ``meta-policy.py`` - classes and functions implementing the meta-policy
6. ``policy_prior.py`` - classes and functions implementing the prior over policies

The main implementation of the **POTMMCP** algorithm is contained in the ``potmmcp/tree/policy.py`` file.

experiments
```````````

This directory contains scripts for running the experiments in each environment as well as Jupyter notebooks for analysing the results and the actual results files.


Results
-------

If you run any of the experiment scripts, by default experiment results are saved to the ``~/potmmcp_results`` directory.


Questions or Issues
-------------------

If you have any questions or issues please email jonathon.schwartz@anu.edu.au or create an issue in the issue section on github.


Authors
-------

- Jonathon Schwartz (primary author)
- Hanna Kurniwati
- Marcus Hutter

Please Cite
-----------

If you use the code in this repository or the **POTMMCP** algorithm, consider citing:

.. code-block:: bibtex
    
    @article{schwartz2023combining,
      title = {Combining a Meta-Policy and Monte-Carlo Planning for Scalable Type-Based Reasoning in Partially Observable Environments}, 
      author = {Jonathon Schwartz and Hanna Kurniawati and Marcus Hutter},
      year = {2023},
      journal = {arXiv preprint arXiv:2306.06067},
      eprint = {2306.06067},
      eprinttype = {arxiv},
      archivePrefix = {arXiv}
    }