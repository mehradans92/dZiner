# dZiner

<!--start-intro-->

![Tests](https://github.com/github/docs/actions/workflows/test.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)



An agentic framework for rational inverse design of materials by replicating and incorporating the expertise of human domain experts across various design tasks and target properties.

<img src="https://github.com/user-attachments/assets/690d1732-a5fe-4ada-aa12-b2bba01eb723" width="1000">

## Human-in-the-loop Inverse Design

Collaborative efforts between a human expert and AI agents hold significant promise. In the case of molecular design for WDR5 ligands, we examined human guidance to refine the modifications based on docking scores and structural generation.

https://github.com/user-attachments/assets/01c067c6-07b4-4efd-94ec-79057f9a2bcf


## Closed-loop Inverse Design

We applied dZiner to the rational inverse design of likely synthesizable organic linkers for metal-organic frameworks with high CO2 adsorption capacity at 0.5 bar of pressure. These MOFs come with pcu topology and three types of inorganic nodes: Cu paddlewheel, Zn paddlewheel, and Zn tetramer (three most frequent node-topology pairs in the hMOF dataset). Design constraints such as keeping molecular weight lower than 600 g/mol and excluding certain potentially unstable functional groups (nitrosylated, chloro-, fluoro-, or trifluoromethyl amines) are simply added to the model as natural language text. 

<img src="https://github.com/user-attachments/assets/efd29d20-4dd4-443e-a409-e0209804edae" width="1000">

<!--end-intro-->


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

## Adding an API Key

If you are using closed-source LLMs from OpenAI or Anthropic, you will need to have a valid `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. If you are on a windows machine you can add keys in your Environment Variables. For linux systems set the key by adding this line to `~/.bashrc`:

```bash
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
```
If you are using the jupyter notebook examples in this repo you may need to restart your kernel after adding the key to your environment
