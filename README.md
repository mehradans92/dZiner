# dZiner

<!--start-intro-->

![Tests](https://github.com/github/docs/actions/workflows/test.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/dziner.svg)](https://badge.fury.io/py/dziner)



An agentic framework for rational inverse design of materials by replicating and incorporating the expertise of human domain experts across various design tasks and target properties.

<img src="https://github.com/user-attachments/assets/690d1732-a5fe-4ada-aa12-b2bba01eb723" width="1000">
The model starts by inputting the material's initial structure as a textual representation. The AI agent dynamically retrieves design guidelines for Property X from scientific literature, the Internet or other resources. Based on these domain-knowledge of guidelines, and any additional design constraints provided in <strong>natural language</strong>, the agent proposes a new candidate and assesses its chemical feasibility in real-time. Next, it estimates Property X for the new candidate, incorporating epistemic uncertainty, using a cost-efficient surrogate model. Optionally, as part of a human-in-the-loop process, the human chemist can review the agent's new candidates and chain-of-thoughts, providing feedback and suggesting further modifications or constraints, creating an opportunity for human-AI collaboration to guide the exploration process. The agent continues exploring the chemical space, guided by chemistry-informed rules, until it meets the convergence criteria.

## Human-in-the-loop Inverse Design

Collaborative efforts between a human expert and AI agents hold significant promise. In the case of molecular design for WDR5 ligands, we examined human guidance to refine the modifications based on docking scores and structural generation.

https://github.com/user-attachments/assets/01ee84a4-0357-4b9c-9305-2258304340a2

## Closed-loop Inverse Design

We applied dZiner to the rational inverse design of likely synthesizable organic linkers for metal-organic frameworks with high CO2 adsorption capacity at 0.5 bar of pressure. These MOFs come with pcu topology and three types of inorganic nodes: Cu paddlewheel, Zn paddlewheel, and Zn tetramer (three most frequent node-topology pairs in the hMOF dataset). Design constraints such as keeping molecular weight lower than 600 g/mol and excluding certain potentially unstable functional groups (nitrosylated, chloro-, fluoro- amines) are simply added to the model as natural language text. 

<img src="https://github.com/user-attachments/assets/3480806e-43b5-423a-8322-5956a2d0c359" width="1000">

## How Can I Use dZiner for My Own Materials Inverse Design Problem?

dZiner can work with different textual representation for materials. You can even apply your own surrogate model to your own materials inverse design problem. Here are some example notebooks that can help you get started:
- [Quick Start](https://mehradans92.github.io/dZiner/quick_start.html)
- [Example 1 - MOF Organic Linker Design for CO2 Adsorption](https://mehradans92.github.io/dZiner/Co2_adorption_MOF.html)
- [Example 2 - Desigining Peptides with a Lower Hemolytic Activity](https://mehradans92.github.io/dZiner/peptide-hemolytic.html)
- [Example 3 - Low CMC Surfactant Design](https://mehradans92.github.io/dZiner/surfactant-cmc.html)
  
<!--end-intro-->


# Installation

Note that with GPU installation, the wall-time of the domain-knowledge tool that uses RAG becomes much shorter.

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
