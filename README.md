# Deita
<p align="center">
  :hugs: <a href="https://huggingface.co/collections/hkust-nlp/deita-6569c198c174808d94cf5bd4">HF Repo</a>&nbsp;&nbsp;&nbsp;
  :page_with_curl: <a href="">Paper</a>
</p>

<p align="center">
  <img src="http://img.peter-we.com//image/20231201214146.png" width="250">
</p>

Welcome to Deita (**D**ata-**E**fficient **I**nstruction **T**uning for **A**lignment) Project! 

This is the **preview version** of Deita. We will continue to update, please stay tuned!


## What is Deita?
Deita is an open-sourced project designed to facilitate **Automatic Data Selection** for instruction tuning in Large Language Models (LLMs), in the siprit of [LIMA](https://arxiv.org/abs/2305.11206).

It concludes:
- **Open-sourced Toolkits** for Automatic Data Selection for Alignment
- **Deita Datasets**: A Series of Concise, High-quality Alignment Data
- **Deita Models**: A Series of Powerful Models on Par with State-Of-The-Art (SOTA) chat LLMs with an Extremely Efficient Instruction Tuning Process. Deita models can be obained by training with just 10x less instruction tuning data compared with other SOTA LLMs

### How far can Deita go?
:bell: Still curious about how far a small amount of high-quality data can lead LLMs? 

Deita may provide an answer for you:

We train our Deita models based on different open-sourced backbone Mistral-7B, LLaMA-2-13B, LLaMA-1-13B using the **automatically selected 6K, 10K** data by our toolkits and **randomly selected 10K** preference data.

| Model                                          | Align     | Data Size  | MT-Bench | AlpacaEval(%) | OpenLLM (Avg.) |
|------------------------------------------------|-----------|------------|----------|---------------|----------------|
| **Proprietary Models**                         |           |            |          |               |                |
| GPT-4-Turbo                                    | ?         | --         | 9.32     | 97.70         | --             |
| GPT-4                                          | SFT + PPO | --         | 8.99     | 95.03         | --             |
| Claude-2                                       | SFT + PPO | --         | 8.06     | 91.36         | --             |
| GPT-3.5-turbo                                  | SFT + PPO | --         | 7.94     | 89.37         | --             |
| **Open-sourced Models based on Mistral-7B**    |           |            |          |               |                |
| Mistral-7B-Instruct-v0.1                       | --        | --         | 6.84     | 69.65         | 60.45          |
| Zephyr-7B-sft                                  | SFT       | 200K       | 5.32     | 75.12         | 60.93          |
| $\text{Zephyr-7B-}\beta$                       | SFT + DPO | 200K + 60K | 7.34     | 90.60         | 66.36          |
| Random                                         | SFT       | 10K        | 5.89     | 56.90         | 61.72          |
| $\text{DEITA}_{10K}$                           | SFT       | 10K        | 7.30     | 80.59         | 64.22          |
| **$\text{DEITA}_{6K}\text{+DPO}$**             | SFT + DPO | 6K + 10K   | 7.46     | 89.32         | xx.xx          |  
| **Open-sourced Models based on LLaMA-2-13B**   |           |            |          |               |                |
| LLaMA2-13B-Chat                                | SFT + PPO | --         | 6.65     | 81.09         | --             |
| Vicuna-13B-v1.5                                | SFT       | 125K       | 6.57     | 78.80         | 61.63          |
| Random                                         | SFT       | 10K        | 5.78     | 65.19         | 61.32          |
| $\text{DEITA}_{10K}$                           | SFT       | 10K        | 6.79     | 81.09         | 62.71          |
| **Open-sourced Models based on LLaMA-1-13B**   |           |            |          |               |                |
| LIMA                                           | SFT       | 1K         | 4.29     | 41.98         | 59.82          |
| WizardLM-13B                                   | SFT       | 70K        | 6.35     | 75.31         | 58.96          |
| Vicuna-13B-v1.3                                | SFT       | 125K       | 6.39     | 82.11         | 60.01          |
| Random                                         | SFT       | 10K        | 6.03     | 71.52         | 60.14          |
| $\text{DEITA}_{10K}$                           | SFT       | 10K        | 6.60     | 78.01         | 64.27          |

We conduct an evaluation on MT-bench, AlpacaEval, OpenLLM Benchmark (ARC, HellaSwag, MMLU, TruthfulQA). We found Deita could compete with SOTA aligned models **with very limited training data**. 

:fire: To our surprise, the Mistral-based $\text{Deita}_{6K}$ model, trained with SFT on 6K data selected by our toolkit, achieved amazing performance by merely undergoing DPO with 10K preference data **randomly sampled** from Ultrafeedback. Specifically, it demonstrated a remarkable MT-Bench score of **7.46** and an AlpacaEval win rate of **89.32**

### :magic\_wand: How can Deita achieve that?
In Deita's 1.0 release, we utilize an innovative approach for data selection, focusing on three key dimensions: complexity, quality, and diversity. To effectively gauge complexity and quality, we've introduced two new metrics, **Evol-Complexity** and **Evol-Quality**. Additionally, our simple yet effective method, **Repr Filter**, is specifically designed to guarantee the diversity of the data we select. By integrating these three critical aspects, we've developed a comprehensive toolkit for automated data selection, ensuring that Deita operates with the most effective and varied datasets. More details please refer to our [paper]()

<p align="center">
  <img src="http://img.peter-we.com//image/20231202224856.png" width="1000">
</p>

### :muscle: What's more?

This is the preview version of Deita project. We will continue to update including

- [ ] Complete data selection pipeline with efficient implementation
- [ ] Our training codebase including training for scorer, SFT, DPO, etc.
- [ ] More automatic data selection strategies
- [ ] CLI-Interface Supported

## News
- :fire: [12/2023] **Deita Models**: ...
- :fire: [12/2023] **Deita Datasets**: Our selected high-quality datasets for instruction tuning, **Deita-6k-v0** [\[🤗 HF Repo\]](https://huggingface.co/datasets/hkust-nlp/Deita-10k-v0) and **Deita-10k-v0** [\[🤗 HF Repo\]](https://huggingface.co/datasets/hkust-nlp/Deita-10k-v0), have been released
- :fire: [12/2023] **Deita Scorers**: Our complexity scorer **Deita-Complexity-Scorer** [\[🤗 HF Repo\]](https://huggingface.co/hkust-nlp/Deita-Complexity-Scorer) and quality scorer **Deita-Quality-Scorer** [\[🤗 HF Repo\]](https://huggingface.co/hkust-nlp/Deita-Quality-Scorer) have been released.

## Contents
  - [🪄 Deita Scorer](#magic\_wand-deita-scorers)
  - [:rocket: Deita Models Checkpoints](#rocket-deita-models)
  - [📚 Deita Dataset](#books-deita-datasets)
## 🪄 Deita Scorers

## :rocket: Deita Models

## :books: Deita Datasets

## Citations
