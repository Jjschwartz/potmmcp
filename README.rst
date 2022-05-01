BA-POSGMCP
###########

Bayes Adaptive Monte-Carlo Planning algorithm for Partially Observable Stochastic Games


Installation
------------

Installing this library requires installing three main dependencies:

1. Pytorch
2. Rllib (version 1.12)
3. baposgmcp (this projec)
4. posggym

Both ``pytorch`` and ``rllib`` will need to be installed before installing ``baposgmcp``. Instructions for installing these can be found on the relevant websites: https://pytorch.org and https://docs.ray.io/en/master/rllib/.

**Note** Make sure to install ``rllib`` version ``1.12.0``. Using ``pip`` this can be installed with the following command:

.. code-block:: bash

   pip install "ray[rllib]==1.12"


After ``pytorch`` and ``rllib`` are both installed, to install ``baposgmcp`` you must first clone the git repo then from the root repo directory (called ``ba-posgmcp``) install using ``pip``:

.. code-block:: bash

   pip install -e .

``posggym`` can be installed by first cloning the repo from https://github.com/RDLLab/posggym and then installing locally as above.


And voila.
