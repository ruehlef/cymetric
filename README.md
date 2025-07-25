# cymetric

![CYMetric plots](/assets/plots.jpg)

cymetric is a Python package for learning of moduli-dependent Calabi-Yau metrics
using neural networks implemented in TensorFlow. This repository contains an updated version that works with tensorflow 2 and Mathematica 13 or older. It also contains a port to pytorch, which was auto-generated with GitHub copilot. The torch implementation is slower than the tensorflow implementation, since some functions could be XLA-compiled in tensorflow, but the torch-compiler did not support the necessary operations.

## Features

The current version is an alpha-release.


## Installation
This guide assumes that you have a working Python 3 (preferably python 3.7 or above) installation (and Sage and Mathematica, if you want to use these features as well). So running ```python3``` should work on your system. Moreover, it assumes that you have installed git. Note that both are standard on Mac and most Linux distributions. For Windows, you will typically have to install them and make sure that for example Python works correctly with Mathematica if you are planing on using the Mathematica interface.

### 1. Install it with Python
If you want to use any existing python installation (note that we recommend using a virtual environment, see below), just run in a terminal
```console
pip install git+https://github.com/ruehlef/cymetric.git
```
This will automatically:
- ✅ Install PyTorch on Python 3.8+ 
- ✅ Install TensorFlow on Python 3.8-3.12
- ⚠️  Skip TensorFlow on Python 3.13+ (not yet supported)
- 📦 Always install core dependencies

To run the example notebooks, you need jupyter. You can install it with
```console
pip install jupyter notebook
```

### 2. Install with virtual environment
#### Using standard virtual environment
Create a new virtual environment in a terminal with

```console
python3 -m venv ~/cymetric
```

Then install with pip directly from github 

```console
source ~/cymetric/bin/activate
pip install --upgrade pip
pip install git+https://github.com/ruehlef/cymetric.git
pip install jupyter notebook
python -m ipykernel install --user --name=cymetric
```

#### Using anaconda
Create a new environment with

```console
conda create -n cymetric python=3.9
```

Then install with pip directly from github 

```console
conda activate cymetric
pip install git+https://github.com/ruehlef/cymetric.git
```

### 3. Install within Sage
Since sage comes with python, all you need to do is run 
```console
pip install git+https://github.com/ruehlef/cymetric.git
```
from within a sage notebook. If you'd rather keep ML and sage separate, you can just install the package (without tensorflow etc.) using 
```console
pip install --no-dependencies git+https://github.com/ruehlef/cymetric.git
```
Then you can use the function ```prepare_toric_cy_data(tv, "toric_data.pickle"))``` to create and store all the toric data needed, and then run the ML algorithms with this data file from a separate package installation (with tensorflow).

### 4. Install within Mathematica
The whole installation process is fully automatic in the [Mathematica notebook](/notebooks/4.Mathematica_integration_example.nb). Just download it and follow the instructions in the notebook. In a nutshell, you run
```console
Get["https://raw.githubusercontent.com/ruehlef/cymetric/main/cymetric/wolfram/cymetric.m"];
PathToVenv = FileNameJoin[{$HomeDirectory, "cymetric"}];
python = Setup[PathToVenv];
```
You can also use an already existing installation. To do so, you run
```console
Get["https://raw.githubusercontent.com/ruehlef/cymetric/main/cymetric/wolfram/cymetric.m"];
PathToVenv = FileNameJoin[{$HomeDirectory, "cymetric"}];
ChangeSetting["Python", PathToVenv]
python = Setup[PathToVenv];
```
Note that this will create a .m file (in the same folder and with the same name as the mathematica notebook) which stores the location of the virtual environment. If you delete this file, mathematica will install a new virtual environment the next time you call ```Setup[PathToVenv]```.

## Tutorials
Once you have installed the package (either in python, or in sage, or in Mathematica), you are probably looking for some examples on how to use it. We provide some tutorials/examples for each case. Just download the example file somewhere on your computer and open it in jupyter. If you created a virtual environment as explained above, you can simply open a terminal and type
```console
jupyter notebook
```
This will open jupyter in your web browser. Navigate to the folder where you downloaded the files and click on them to open.

0. In [1.PointGenerator.ipynb](notebooks/0.GettingStarted.ipynb) you get a pipeline to generate points and learn the metric for the Quintic, both in pytorch and tensorflow.
1. In [1.PointGenerator.ipynb](notebooks/1.PointGenerator.ipynb) we explore the different PointGenerators for codimension-1 CICY, general CICYs and CY in toric varieties on the Fermat Quintic. 
2. In [2.TensorFlow_models.ipynb](notebooks/2.TensorFlow_models.ipynb) we explore some of the TF custom models with the data generated in the first notebook. 
3. In [3.Sage_integration_.ipynb](notebooks/3.Sage_integration_example.ipynb) we illustrate how to run the package from within Sage to compute the CY metric on a Kreuzer-Skarke model.
4. In [Mathematica_integration_example.nb](/notebooks/4.Mathematica_integration_example.nb), we illustrate how to call the PointGenerators and the models for training and evaluation. Furthermore, there are arbitrary precision PointGenerators based on the wolfram language.

## Cymetric Framework Selection

This document describes how to control which framework (PyTorch or TensorFlow) is used by the cymetric compatibility layer.

### Default Behavior

When both PyTorch and TensorFlow are installed, cymetric defaults to the faster **TensorFlow**:

```python
import cymetric
from cymetric.models.measures import ricci_measure
print(ricci_measure.__module__)  # cymetric.tensorflow.models.measures
```

### Framework Selection Methods

#### Method 1: Environment Variable (Recommended)

Set the `CYMETRIC_FRAMEWORK` environment variable before importing cymetric:

```bash
export CYMETRIC_FRAMEWORK=torch
python your_script.py
```

Or in Python:
```python
import os
os.environ['CYMETRIC_FRAMEWORK'] = 'torch'  # Must be before importing cymetric
import cymetric
from cymetric.models.measures import ricci_measure
```

Valid values: `'torch'`, `'pytorch'`, `'tf'`, `'tensorflow'`

#### Method 2: Runtime Switching

Change the framework after importing cymetric:

```python
import cymetric
from cymetric.models.measures import ricci_measure  # Uses default (TensorFlow)

# Switch to PyTorch
cymetric.set_preferred_framework('torch')
from cymetric.models.measures import ricci_measure  # Now uses PyTorch
```

#### Method 3: Check Current Framework

```python
import cymetric
print(f"Available frameworks: PyTorch={cymetric.TORCH_AVAILABLE}, TensorFlow={cymetric.TENSORFLOW_AVAILABLE}")
print(f"Currently using: {cymetric.PREFERRED_FRAMEWORK}")
```

### Direct Framework Access

You can always import from specific frameworks directly:

```python
# Always use PyTorch
from cymetric.torch.models.measures import ricci_measure

# Always use TensorFlow  
from cymetric.tensorflow.models.measures import ricci_measure
```

### Compatibility Layer Modules

The following modules support automatic framework selection:

- `cymetric.models.measures`
- `cymetric.models.callbacks`
- `cymetric.models.losses`
- `cymetric.models.metrics`
- `cymetric.models.fubinistudy`
- `cymetric.models.torchmodels` / `cymetric.models.tfmodels`
- `cymetric.models.torchhelper` / `cymetric.models.tfhelper`

All of these automatically redirect to the appropriate framework implementation.


## Conventions and normalizations
We summarize the mathematical conventions we use in [this .pdf file](./assets/conventions.pdf).

## Contributing

We welcome contributions to the project. Those can be bug reports or new features, 
that you have or want to be implemented.

## Citation

You can find our paper on the [arXiv](https://arxiv.org/abs/2111.01436). It was presented at the [ML4PS workshop](https://ml4physicalsciences.github.io/2021/) of [NeurIPS 2021](https://neurips.cc/Conferences/2021/Schedule?showEvent=21862). If you find this package useful in your work, cite the following bib entry:

```
@article{Larfors:2021pbb,
    author = "Larfors, Magdalena and Lukas, Andre and Ruehle, Fabian and Schneider, Robin",
    title = "{Learning Size and Shape of Calabi-Yau Spaces}",
    eprint = "2111.01436",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "UUITP-53/21",
    year = "2021",
    journal = "Machine Learning and the Physical Sciences, Workshop at 35th NeurIPS",
}
```
