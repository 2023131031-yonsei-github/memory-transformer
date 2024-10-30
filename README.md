# Repository of Memory-Transformer Experiment
This is the repository of memory-transformer experiment.

## Requirements

    torch
    fire
    sentencepiece
    datasets

If you want to train Memory-Transformer with official Llama implemetation from Facebook Llama repository, you need to install,

    fairscale
Note that you need to edit model.py accordingly for backward() function in torch - See memory_transformer/llama/model.py.

## Installation
To install, move setup.py and requirements.txt outside of the directory:
    
    ./
        - setup.py
        - requirements.txt
        memory-transformer/
            ...

and run the command below.

    pip install -e .
It will automatically install memory_transformer package and every requirements except fairscale.

CAUTION : Installing package with -e have some known issues.

## Description of Files

    download.sh : a shell for downloading Llama. Not available at current version.
## Cautions
You need to install Llama3.2-1B or Llama3.2-1B-Instruct manually to run train.py.

Running train.py will automatically download BookCorpus dataset from HuggingFace by datasets package.

## Citations
