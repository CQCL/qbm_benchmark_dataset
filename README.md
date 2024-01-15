# Training of Quantum Boltzmann Machines

Library for training QBMs based on [`quimb`](https://quimb.readthedocs.io/en/latest/index.html), a popular python library for quantum information and many-body physics, with MPI acceleration and Tensor Network methods.

## Setup

The package environment is handled by [`poetry`](https://python-poetry.org/docs/) which will install all the dependencies and the package when running `poetry install` in the root folder of the project.

## Benchmark

The benchmarking script can be run with `poetry run python scripts/benchmarking.py`. A list of command line arguments is shown below:

```bash
usage: benchmark.py [-h] [--n N] [--t T] [--b B] [--l L] [--dn DN] [--lr LR] [--e E] [--er ER] [--sn SN] [--qre] [--pre_l PRE_L] [--pre_lr PRE_LR] [--pre_e PRE_E] [--seed SEED] [--output OUTPUT]

Train a QBM model to represent a target Gibbs state

options:
  -h, --help       show this help message and exit
  --n N            Number of qubits (4)
  --t T            Label of target model (0)
  --b B            Inverse temperature of target model (1.0)
  --l L            Label of QBM model (0)
  --dn DN          Intensity of depolarizing noise (0.0)
  --lr LR          Learning rate (0.01)
  --e E            Number of traninig epochs (1000)
  --er ER          Error tolerance for gradients (1e-6)
  --sn SN          Standard deviation of gaussian shot noise for computing gradients (0.0)
  --qre            If we want to compute and output relative entropies
  --pre_l PRE_L    Label of QBM model for pretraining (None)
  --pre_lr PRE_LR  Learning rate for pretraining (0.01)
  --pre_e PRE_E    Number of traninig epochs for pretraining (300)
  --seed SEED      Seed for PRNG (1)
  --output OUTPUT  Output for data and figures (data/)
```
