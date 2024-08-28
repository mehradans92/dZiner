[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# dZiner

Chemist AI Agent for Materials Inverse Design

<img src="https://github.com/mehradans92/dZiner/assets/51170839/68d9495f-87fb-4697-89ea-cc3c553b96f2" width="500">



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
## Adding an API Key

You need to have a valid OPENAI API key. If you are on a windows machine you can add `OPENAI_API_KEY` in your Environment Variables. For linux systems set the key by adding this line to `~/.bashrc`:

```bash
export OPENAI_API_KEY=your_api_key_here
```
If you are using the jupyter notebook examples in this repo you may need to restart your kernel after adding the key to your environment

### Surrogate Models

Make sure you follow additional installation of packages for different surrogate models. Requirements can be found in each folder.

```bash
cd surrogates/YOUR_MODEL_OF_INTEREST
pip install -r requirements.txt
```
