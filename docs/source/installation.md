# Installation

Note that with GPU installation, the wall-time of the domain-knowledge tool that uses RAG becomes much shorter. After this installation, you wiil need to install the surrogate model you want to use (see below).

## Installation - `pip`(GPU)

```bash
pip install dziner
conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.8.0
python -m ipykernel install --user --name dziner --display-name "dziner"
```

## Installation - `pip`(CPU)

```bash
pip install dziner
conda install -c pytorch/label/nightly faiss-cpu
python -m ipykernel install --user --name dziner --display-name "dziner"
```

## Installation - dev mode (GPU)

You can clone the source code and install in developer mode:

```bash
conda create -n dziner python=3.11.9
conda activate dziner

git clone https://github.com/mehradans92/dziner.git && cd dziner
pip install -e .
conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.8.0
python -m ipykernel install --user --name dziner --display-name "dziner"
```

## Installation - dev mode (CPU)

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

## Adding an API Key

If you are using closed-source LLMs from OpenAI or Anthropic, you will need to have a valid `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. If you are on a windows machine you can add keys in your Environment Variables. For linux systems set the key by adding this line to `~/.bashrc`:

```bash
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
```
If you are using the jupyter notebook examples in this repo you may need to restart your kernel after adding the key to your environment
