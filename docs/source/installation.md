# Installation

## Installation (GPU)

You can clone the source code and install in developer mode:

```bash
conda create -n dziner python=3.11.9
conda activate dziner

git clone https://github.com/mehradans92/dziner.git && cd dziner
pip install -e .
conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.8.0
python -m ipykernel install --user --name dziner --display-name "dziner"
```

## Installation (CPU)

If you do not have a gpu in your machine (OSX for example) you will need to execute the following instead:

```bash
conda create -n dziner python=3.11.9
conda activate dziner

git clone https://github.com/mehradans92/dziner.git && cd dziner
pip install -e .
conda install -c pytorch/label/nightly faiss-cpu
python -m ipykernel install --user --name dziner --display-name "dziner"
```
## Surrogate Models

Make sure you follow additional installation of packages for different surrogate models. Requirements can be found in each folder.

```bash
cd dziner/surrogates/YOUR_MODEL_OF_INTEREST
pip install -r requirements.txt
```