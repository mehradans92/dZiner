# dZiner

<!--start-intro-->

![Tests](https://github.com/github/docs/actions/workflows/test.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)



An agentic framework for rational inverse design of materials by replicating and incorporating the expertise of human domain experts across various design tasks and target properties.

<img src="https://github.com/user-attachments/assets/690d1732-a5fe-4ada-aa12-b2bba01eb723" width="1000">

<!--end-intro-->

.. include:: docs/source/installation.md

## Adding an API Key

If you are using closed-source LLMs from OpenAI or Anthropic, you will need to have a valid `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. If you are on a windows machine you can add keys in your Environment Variables. For linux systems set the key by adding this line to `~/.bashrc`:

```bash
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
```
If you are using the jupyter notebook examples in this repo you may need to restart your kernel after adding the key to your environment
