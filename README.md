# Deita

<p align="center">
  <img src="./assets/logo-final.png" width="800">
</p>

<p align="center">
  :hugs: <a href="https://huggingface.co/collections/hkust-nlp/deita-6569c198c174808d94cf5bd4">HF Repo</a>&nbsp;&nbsp;&nbsp;
  :page_with_curl: <a href="">Paper</a>
</p>

Welcome to Deita (**D**ata-**E**fficient **I**nstruction **T**uning for **A**lignment) Project! 

This is the **preview version** of Deita. We will continue to update, please stay tuned!


## What is Deita?
Deita is an open-sourced project designed to facilitate **Automatic Data Selection** for instruction tuning in Large Language Models (LLMs).

It includes:
- **Open-sourced Toolkits** for automatic data selection in instruction tuning
- **Deita Datasets**: A series of extremely *lightweight*, high-quality alignment SFT data. We release 6k-sized and 10k-sized datasets in the first release
- **Deita Models**: A series of powerful models on par with SOTA chat LLMs with an extremely efficient instruction tuning Process. Deita models can be obained by training with 10x less instruction tuning data compared with other SOTA LLMs

## News

- :fire: [12/2023] We release the first collection of the Deita resources [here](https://huggingface.co/collections/hkust-nlp/deita-6569c198c174808d94cf5bd4), which include a series of extremely lightweight, effective sft datasets, the data complexity/quality scorer models, as well as the resulted deita chat models. 

## How far can Deita go?
:bell: Still curious about how far a small amount of high-quality data can lead LLMs? 

Deita may provide an answer for you:

**🔦 Highlights**
| Model                                          | Align        | Data Size  | MT-Bench | AlpacaEval(%) |
|------------------------------------------------|--------------|------------|----------|---------------|
| Zephyr-7B-sft                                  | SFT          | 200K       | 5.32     | 75.12         |
| Zephyr-7B-$\beta$                      | SFT + DPO    | 200K SFT + 60K DPO | 7.34     | 90.60         |
| OpenChat-3.5                                   | C-RLFT | >70K C-RLFT | 7.81     | 88.51         |
| Starling-7B                                    | C-RLFT + APA | >70K C-RLFT + 183K APA | 8.09     | 91.99         |
| Tulu-2-13B                                     | SFT          | 326K       | 6.70     | 78.90         |
| Tulu-2-13B+DPO                                 | SFT + DPO    | 326K SFT + 60K DPO | 7.00     | 89.50         |
| LLaMA2-13B-Chat                                | SFT + PPO    | --         | 6.65     | 81.09         |
| WizardLM-13B-v1.2                              | SFT          | >70K       | 7.09     | 89.17         |
| Vicuna-13B-v1.5                                | SFT          | >125K      | 6.57    | 78.80         |
| DEITA-7B-v1.0 (6K)          | SFT          | 6K       |   7.17   |    80.95      |
| DEITA-7B-v1.0-sft            | SFT          | 10K        | 7.30     | 80.59         |
| DEITA-7B-v1.0 | SFT + DPO    | 6K SFT + 10K DPO | 7.46     | 89.32         |

DEITA models are based on Mistral-7B-v0.1. :fire: To our surprise, the Mistral-based DEITA-7B-v1.0 model, trained with SFT on 6K data selected by our toolkit, achieved amazing performance by merely undergoing DPO with 10K preference data **randomly sampled** from Ultrafeedback. Specifically, it demonstrated a remarkable MT-Bench score of **7.46** and an AlpacaEval win rate of **89.32**

Please refer to [this table](#chart\_with\_upwards\_trend-full-evaluations) for full evaluations, which includes DEITA models with LLaMA base models and comparisons with other data selection approaches.



## :chart_with_upwards_trend: Full Evaluations

<details>
  <summary>See full evaluations</summary>

  | Model                                          | Align     | Data Size  | MT-Bench | AlpacaEval(%) | OpenLLM (Avg.) |
|------------------------------------------------|-----------|------------|----------|---------------|----------------|
| **Proprietary Models**                         |           |            |          |               |                |
| GPT-4-Turbo                                    | ?         | --         | 9.32     | 97.70         | --             |
| GPT-4                                          | SFT + PPO | --         | 8.99     | 95.03         | --             |
| Claude-2                                       | SFT + PPO | --         | 8.06     | 91.36         | --             |
| GPT-3.5-turbo                                  | SFT + PPO | --         | 7.94     | 89.37         | --             |
| **Open-sourced Models based on Mistral-7B**    |           |            |          |               |                |
| Mistral-7B-Instruct-v0.1                       | --        | --         | 6.84     | 69.65         | 60.45          |
| Zephyr-7B-sft                                  | SFT       | 200K SFT      | 5.32     | 75.12         | 60.93          |
| $\text{Zephyr-7B-}\beta$                       | SFT + DPO | 200K SFT + 60K DPO | 7.34     | 90.60         | 66.36          |
| OpenChat-3.5                                   | C-RLFT | >70K C-RLFT | 7.81     | 88.51         | --           |
| Starling-7B                                    | C-RLFT + APA | >70K C-RLFT + 183K APA | 8.09     | 91.99         | --            |
| Random                                         | SFT       | 10K SFT       | 5.89     | 56.90         | 61.72          |
| DEITA-7B-v1.0-sft (6K)                           | SFT       | 6K SFT       | 7.17     | 80.95         | --          |
| DEITA-7B-v1.0-sft                           | SFT       | 10K SFT       | 7.30     | 80.59         | 64.22          |
| DEITA-7B-v1.0             | SFT + DPO | 6K SFT + 10K DPO   | 7.46     | 89.32         | xx.xx          |  
| **Open-sourced Models based on LLaMA-2-13B**   |           |            |          |               |                |
| Tulu-2-13B                                     | SFT       | 326K SFT      | 6.70     | 78.90         | --             |
| Tulu-2-13B+DPO                                 | SFT + DPO | 326K SFT + 60K DPO | 7.00     | 89.50         | --             |
| LLaMA2-13B-Chat                                | SFT + PPO | --         | 6.65     | 81.09         | --             |
| WizardLM-13B-v1.2                              | SFT          | >70K SFT      | 7.09     | 89.17         | --             |
| Vicuna-13B-v1.5                                | SFT       | 125K SFT      | 6.57     | 78.80         | 61.63          |
| Random                                         | SFT       | 10K SFT       | 5.78     | 65.19         | 61.32          |
| DEITA-LLaMA2-13B-v1.0-sft                           | SFT       | 10K SFT       | 6.79     | 81.09         | 62.71          |
| **Open-sourced Models based on LLaMA-1-13B**   |           |            |          |               |                |
| LIMA                                           | SFT       | 1K SFT        | 4.29     | 41.98         | 59.82          |
| WizardLM-13B                                   | SFT       | 70K SFT       | 6.35     | 75.31         | 58.96          |
| Vicuna-13B-v1.3                                | SFT       | 125K SFT      | 6.39     | 82.11         | 60.01          |
| Random                                         | SFT       | 10K SFT       | 6.03     | 71.52         | 60.14          |
| DEITA-LLaMA1-13B-v1.0-sft                           | SFT       | 10K SFT       | 6.60     | 78.01         | 64.27          |

</details>

## :rocket: Deita Resources

| Resource                                       | Link     | License  |
|------------------------------------------------|-----------|------------|
| **Deita Datasets**                                   |           |            |
| deita-6k-v0                                    | [:hugs: HF Repo](https://huggingface.co/datasets/hkust-nlp/deita-6k-v0)          | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| deita-10k-v0                                    | [:hugs: HF Repo](https://huggingface.co/datasets/hkust-nlp/deita-10k-v0)          | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| **Scorers**                                   |           |             |
|  deita-complexity-scorer                      | [:hugs: HF Repo](https://huggingface.co/hkust-nlp/deita-complexity-scorer)          | [LLaMA License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)|
|  deita-quality-scorer               | [:hugs: HF Repo](https://huggingface.co/hkust-nlp/deita-quality-scorer)          | [LLaMA License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)|
| **Deita Models**                                   |           |             |
| DEITA-7B-v1.0-sft (6K)                | [:hugs: HF Repo]()           | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)             | 
| DEITA-7B-v1.0-sft                | [:hugs: HF Repo]()           | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)             | 
| DEITA-7B-v1.0                | [:hugs: HF Repo]()           | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)             | 
| DEITA-LLaMA2-13B-v1.0-sft         | [:hugs: HF Repo](https://huggingface.co/hkust-nlp/deita-llama2-13b-v1.0-sft)           |  [LLaMA 2 License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)           |
| DEITA-LLaMA1-13B-v1.0-sft          | [:hugs: HF Repo](https://huggingface.co/hkust-nlp/deita-llama1-13b-v1.0-sft)          |  [LLaMA License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)           |




### :muscle: What's more?

This is the preview version of Deita project. We will continue to update including

- [ ] Release data selection pipeline with efficient implementation
- [ ] More automatic data selection strategies
- [ ] CLI-Interface Supported
- [ ] Online Demo

## Citations›
If you find the content of this project helpful, please cite this repo or our paper as follows:

```
@misc{liu2023what,
      title={What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning}, 
      author={Wei Liu and Weihao Zeng and Keqing He and Yong Jiang and Junxian He},
      year={2023},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@misc{deita2023,
  author = {Wei Liu and Weihao Zeng and Keqing He and Yong Jiang and Junxian He},
  title = {Deita: Data-Efficient Instruction Tuning for Alignment},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hkust-nlp/deita}}
}
```