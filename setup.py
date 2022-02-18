"""BA-POMGMCP package install info """
import sys
from setuptools import setup, find_packages

try:
    import torch
except ImportError:
    print(
        "BA-POSGMCP depends on the pytorch library. For installation "
        "instructions visit https://pytorch.org/"
    )
    sys.exit(1)

try:
    import ray.rllib
except ImportError:
    print(
        "BA-POSGMCP depends on the ray RLlib library. For installation "
        "instructions visit https://docs.ray.io/en/latest/rllib/index.html"
    )
    sys.exit(1)

# Environment and test specific dependencies.
extras = {
    "test": ["pytest>=7.0"],
    "render": ["matplotlib>=3.5"]
}

extras['all'] = [item for group in extras.values() for item in group]


setup(
    name='baposgmcp',
    version='0.0.1',
    url="https://github.com/Jjschwartz/ba-posgmcp",
    description=(
        "Bayes Adaptive Monte-Carlo Planning for Partially Observable "
        "Stochastic Games."
    ),
    author="Jonathon Schwartz",
    author_email="Jonathon.Schwartz@anu.edu.au",
    license="MIT",
    packages=[
        package for package in find_packages()
        if package.startswith('baposgmcp')
    ],
    install_requires=[],
    extras_require=extras,
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
    zip_safe=False
)
