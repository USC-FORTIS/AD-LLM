# AD-LLM: Benchmarking Large Language Models for Anomaly Detection

## Overview

**AD-LLM** introduces the first benchmark evaluates how large language models (LLMs) can help with natural language processing (NLP) in anomaly detection (AD). We examine three key tasks:

1. **Zero-shot Detection** - Leveraging LLMs' pre-trained knowledge to perform AD without task-specific training.

2. **Data Augmentation** 
    1. **Synthetic Data Generation** - Generating synthetic data to improve AD models.
    2. **Category Descriptions Generation** - Generating category descriptions to enhance LLM-based AD.
   
3. **Model Selection** - Recommending suitable unsupervised AD models via LLMs.

Our benchmark evaluates LLMs such as GPT-4o and Llama 3.1 across multiple datasets.

## Environment Set-up
> We use anaconda to create python environment and install required libraries:
```
# creat the environment and activate it
conda create --name ad_llm python=3.11
conda activate ad_llm

# install basic packages
conda install numpy scipy scikit-learn matplotlib tqdm

# please adjust the pytorch installation based on the situation
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

# install PyOD (https://github.com/yzhao062/pyod)
conda install -c conda-forge pyod

# install for Llama
conda install conda-forge::transformers
pip install --upgrade huggingface hub
pip install accelerate

# install for GPT
pip install openai
```

## Usage
### 1. Zero-shot Detection
* Set `_ad_setting=1` in `config.py` to run "Normal Only" setting; set `_ad_setting=2` in `config.py` to run "Normal + Anomaly" setting.
* To run experiments with Llama 3.1: `python ad_llama.py`.
* To run experiments with GPT-4o: `python ad_gpt.py`.
### 2. Data Augmentation
#### 1. Synthetic Data Generation
* Change `_num_keyword_groups_act` in `config.py` to adjust the size of synthetic samples per category.
* To generate synthetic data: `python aug_synth_gpt.py`.
* To run experiments with synthetic data: `python baseline_w_gpt_embed.py`
#### 2. Category Descriptions Generation
* For Llama:
    * To generate category description: `python aug_desc_llama.py`.
    * To run experiments with category description, set `_use_desc = True`, then run `python ad_llama.py`.
* For GPT:
    * To generate category description: `python aug_desc_gpt.py`.
    * To run experiments with category description, set `_use_desc = True`, then run `python ad_gpt.py`.
* Remember to set `_use_desc = False` back when you do not wish to use category description.
### 3. Model Selection
* To run experiments: `python select_gpt.py`.

# Notes
* We provide one example dataset "BBC News". Please check [NLP-ADBench](https://github.com/USC-FORTIS/NLP-ADBench) for more datasets (AG News, IMDB Reviews, N24 News, and SMS Spam) with the same setting.
* 