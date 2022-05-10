# Continual Transformers: Redundancy-Free Attention for Online Inference

<div align="left">

  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" height="20">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" height="20">
  </a>
</div>

Official implementation of _Continual Transformers_ including ready-to-use modules for Continual Inference.

<div align="center">
  <img src="figures/CoReDotProductAttention.png" width="500">
  <br>
  <div align="left">
  Fig. 1: Continual Retroactive Dot-Product Attention. 
  The query (Q), key (K), and value (V) matrices are aggregated over time by caching the step vectors q_n, k_n, and v_n in a FIFO queue. During each step, only the entries of A associated with q_n, k_n, and the oldest K step, k_o are computed. 
  The diagonal entries of the row-normalisation matrix D as well as the AV can be updated retroactively by subtracting features corresponding to k_o and adding features related to k_n to the cached outputs of the previous step, D_{mem} and AV_{mem}, respectively.
  </div>
  <br>
</div>

<div align="center">
  <img src="figures/CoSiDotProductAttention.png" width="500">
  <br>
  <div align="left">
  Fig. 2: Continual Single-Output Dot-Product Attention. 
        The key (K) and value (V) matrices are aggregated over time by caching the step vectors k_n and v_n in a FIFO queue. During each step, only the attention output associated with q is computed.
  </div>
  <br>
</div>


## Setup

Continual Transformers and its modules can be installed using:
```bash
pip install -e .[dev]
```

## Tests
Run unit tests with coverage report:
```bash
python -m pytest --cov continual_transformers --cov-report term-missing 
```


## Experiments and results
The experiment code-base is split into seperate repositories for Online Action Detection and Online Audio Classification, which are supplied as seperate repositories in the supplemental material.