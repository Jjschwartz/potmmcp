POSGGym
#######

Library of Partially Observable Stochastic Game (POSG) environments coupled with dynamic models of each environment.

.. notes::
  This is a partial implementation of the posggym library which contains only the environments and functionality needed for POTMMCP. 

  The full implementation is available at: `https://github.com/RDLLab/posggym/ <https://github.com/RDLLab/posggym/>`_

  Noting that the API of the latest version of posggym may not be compatible with the implementation of POTMMCP.


Installation
------------

At the moment we only support installation by cloning the repo and installing locally.

Once the repo is cloned, you can install POSGGym using PIP by navigating to the `posggym` root directory (the one containing the `setup.py` file), and running:

.. code-block:: bash

   pip install -e .

   # use the following to install all dependencies
   pip install -e .[all]




