# Deita
<p align="center">
  :hugs: <a href="https://huggingface.co/collections/hkust-nlp/deita-6569c198c174808d94cf5bd4">HF Repo</a>&nbsp;&nbsp;&nbsp;
  :page_with_curl: <a href="">Paper</a>
</p>

Welcome to Deita (**D**ata-**E**fficient **I**nstruction **T**uning for **A**lignment) Project!


## What is Deita?
Deita is an open-sourced project designed to facilitate **Automatic Data Selection** for instruction tuning in Large Language Models (LLMs), in the siprit of LIMA[1].

It concludes:
- **Open-sourced Toolkits** for Automatic Data Selection for Alignment
- **Deita Datasets**: A Series of Concise, High-quality Alignment Data
- **Deita Models**: A Series of Powerful Models on Par with State-Of-The-Art (SOTA) chat LLMs with an Extremely Efficient Instruction Tuning Process. Deita models can be obained by training with just 10x less instruction tuning data compared with other SOTA LLMs

:bell: Still curious about how far a small amount of high-quality data can lead LLMs? Deita may provide an answer for you.

| Model                                          | Data Size | MT-Bench | AlpacaEval | OpenLLM Benchmark (Avg.) |
|------------------------------------------------|-----------|----------|------------|-------------------------|
| **Open-sourced Models based on Mistral-7B**    |           |          |            |                         |
| Mistral-7B-Instruct-v0.1                       | --        | 6.84     | 69.65      | 60.45                   |
| Zephyr-7B-sft                                  | 200K      | 5.32     | 75.12      | 60.93                   |
| Random                                         | 10K       | 5.89     | 56.90      | 61.72                   |
| $\text{DEITA}_{10K}$                           | 10K       | **7.29** | **80.59**  | **64.22**               |
| **Open-sourced Models based on LLaMA-2-13B**   |           |          |            |                         |
| Vicuna-13B-v1.5                                | 125K      | 6.57     | 78.80      | 61.63                   |
| Random                                         | 10K       | 5.78     | 65.19      | 61.32                   |
| $\text{DEITA}_{10K}$                           | 10K       | **6.79** | **81.09**  | **62.71**               |
| **Open-sourced Models based on LLaMA-1-13B**   |           |          |            |                         |
| LIMA                                           | 1K        | 4.29     | 41.98      | 59.82                   |
| WizardLM-13B                                   | 70K       | 6.35     | 75.31      | 58.96                   |
| Vicuna-13B-v1.3                                | 125K      | 6.39     | **82.11**  | 60.01                   |
| Random                                         | 10K       | 6.03     | 71.52      | 60.14                   |
| $\text{DEITA}_{10K}$                           | 10K       | **6.60** | 78.01      | **64.27**               |


<!-- :bell: Our model Deita-2-10k-13b through **only supervised fine-tuning (sft)** with **only 10k** data based on LLaMA-2-13B achieves **6.75** on [MT-Bench](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) benchmark, which even outperforms [LLaMA-13B-Chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) with more fancy alignment techniques! -->

<!-- For more details, please refer to our paper: [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning]() -->

## News
...

## Contents

- [Deita](#deita)
  - [What is Deita?](#what-is-deita)
  - [News](#news)
  - [Contents](#contents)
  - [:magic\_wand: Automatic Data Selection](#magic_wand-automatic-data-selection)
    - [Overview](#overview)
    - [Data Scorer](#data-scorer)
    - [Pipelines](#pipelines)
  - [:rocket: Deita-Family](#rocket-deita-family)
  - [Deita Dataset](#deita-dataset)
  - [Deita Models](#deita-models)
  - [TODO](#todo)
  - [Citations](#citations)


## :magic_wand: Automatic Data Selection

### Overview

### Data Scorer

### Pipelines

## :rocket: Deita-Family

## Deita Dataset

## Deita Models

## TODO

- [ ] Deita Pipelines
- [ ] CLI-Interface Supported

## Citations
...
