# Continual Transformers: Redundancy-Free Attention for Online Inference

<div align="left">
  <!-- <a href="https://pypi.org/project/continual-transformers/">
    <img src="https://img.shields.io/pypi/pyversions/continual-transformers" height="20" >
  </a> -->
  <a href="http://arxiv.org/abs/2201.06268">
    <img src="http://img.shields.io/badge/paper-arxiv.2201.06268-B31B1B.svg" height="20" >
  </a>
  <!-- <a href="https://badge.fury.io/py/continual-transformers">
    <img src="https://badge.fury.io/py/continual-transformers.svg" height="20" >
  </a> -->
  <!-- <a href="https://pepy.tech/project/continual-transformers">
    <img src="https://pepy.tech/badge/continual-transformers/month" height="20">
  </a> -->
  <a href="https://codecov.io/gh/LukasHedegaard/continual-transformers">
    <img src="https://codecov.io/gh/LukasHedegaard/continual-transformers/branch/main/graph/badge.svg?token=XW1UQZSEOG"/>
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" height="20">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" height="20">
  </a>
</div>

Official implementation of [Continual Transformers](http://arxiv.org/abs/2201.06268) including ready-to-use modules for [Continual Inference](https://github.com/LukasHedegaard/continual-inference).

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

Continual Transformers and its modules can be installed in in your project using:
```setup
pip install git+https://github.com/LukasHedegaard/continual-transformers.git
```


## Experiments and results
The experiment code-base is split into seperate repositories for [Online Action Detection](https://github.com/LukasHedegaard/CoOadTR) and [Online Audio Classification](https://gitlab.au.dk/maleci/continual-transformer-audio-classification). Below, we present a summary of result from the paper. 

<div align="center">
  <img src="figures/Table6.png" width="500">
</div>


<div align="center">
  <br>
  <img src="figures/Table7.png" width="450">
</div>






## Citation   
```
@article{hedegaard2022cotrans,
  title={Continual Transformers: Redundancy-Free Attention for Online Inference},
  author={Lukas Hedegaard and Alexandros Iosifidis},
  journal={preprint, arXiv:2201.06268},
  year={2022}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
