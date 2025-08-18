<h1 align='center'>Nequix</h1>

See more information in our [preprint](https://teddykoker.com/docs/nequix.pdf).

## Usage

### Installation

```bash
pip install git+https://github.com/atomicarchitects/nequix
```

### ASE calculator

Using `nequix.calculator.NequixCalculator`, you can perform calculations in
ASE with a pre-trained Nequix model.

```python
from nequix.calculator import NequixCalculator

atoms = ...
atoms.calc = NequixCalculator("nequix-mp-1")
```


### Training

Models are trained with the `nequix_train` command using a single `.yml`
configuration file:

```bash
nequix_train <config>.yml
```

To reproduce the training of Nequix-MP-1, first clone the repo and sync the environment:

```bash
git clone https://github.com/atomicarchitects/nequix.git
cd nequix
uv sync
```


Then download the MPtrj data from
https://figshare.com/files/43302033 into `data/` then run the following to extract the data:

```bash
bash data/download_mptrj.sh
```

Then start the training run. The first time this is run it will preprocess the data into HDF5 files:

```bash
nequix_train configs/nequix-mp-1.yml
```

This will take less than 125 hours on a single 4 x A100 node. The `batch_size` in the
config is per-device, so you should be able to run this on any number of GPUs
(although hyperparameters like learning rate are often sensitive to global batch
size, so keep in mind).

