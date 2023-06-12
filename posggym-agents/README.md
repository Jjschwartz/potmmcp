# POSGGym Agents

POSGGym-Agents is a collection of agent policies and policy training code for [POSGGym](https://github.com/RDLLab/posggym) environments. 

> **_NOTE:_** This is a partial implementation of the posggym-agents library which contains only the environments and functionality needed for POTMMCP. The full implementation has been incorporated into the POSGGym library available at <https://github.com/RDLLab/posggym>. Noting that the API of the latest version of posggym may not be compatible with the implementation of POTMMCP.


## Installation

This project depends on the [PyTorch](https://pytorch.org/) and [Ray RLlib](https://docs.ray.io/en/releases-1.12.0/rllib/index.html) libraries. Specifically pytorch version `>= 1.11` and rllib version `1.12`. We recommend install `torch` before installing the POSGGym-Agents package to ensure you get the version of `torch` that works with your CPU/GPU setup.

This version of POSGGym-Agents (version=`0.1.0`) can be installed by:

```bash
cd posggym-agents
pip install -e .
```
