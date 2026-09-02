<h1 align="center">LLM-PeerReview</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2512.23213">
    <img src="https://img.shields.io/badge/arXiv-2512.23213-B31B1B.svg?logo=arXiv" alt="arXiv paper"></a>
  <a href="https://huggingface.co/papers/2512.23213">
    <img src="https://img.shields.io/badge/Hugging%20Face-Paper-FFD21E.svg?logo=huggingface" alt="Hugging Face Paper"></a>
  <a href="https://zeyuji.github.io/LLM-PeerReview/">
    <img src="https://img.shields.io/badge/🌐%20Website-Visit%20Now-purple" alt="Website"></a>
  <a href="https://mp.weixin.qq.com/s/qqR5BW-TkHBaqPA5O-tWXw">
    <img src="https://img.shields.io/badge/📑%20Blog-(Chinese)-orange" alt="Blog (Chinese)"></a>
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/zeyuji/LLM-PeerReview?style=social">
</p>

## ✨ Repository Highlights

> 🔥 **Official Implementation of "Scoring, Reasoning, and Selecting the Best! Ensembling Large Language Models via a Peer-Review Process" (arXiv 2025)**

> 📌 **This repository provides:**
> - 🌀 **Two variants of LLM-PeerReview:** LLM-PeerReview-Average & LLM-PeerReview-Weighted
> - 📖 **Reproductions of recent LLM Ensemble baselines:** Random, GaC, Agent-Forest, Smoothie-Global, Smoothie-Local, Multi-Agent Debate (MAD), SAFE, and CoRE
>   <details>
>   <summary><small>Click to see references</small></summary>
>   
>   - **Random:** [Lu et al., 2024](https://arxiv.org/abs/2401.02994) and [Guha et al., 2024](https://arxiv.org/abs/2412.04692)  
>   - **GaC:** [Yu et al., 2024](https://arxiv.org/abs/2406.12585)  
>   - **Agent-Forest:** [Li et al., 2024](https://arxiv.org/abs/2402.05120)  
>   - **Smoothie:** [Guha et al., 2024](https://arxiv.org/abs/2412.04692)
>   - **MAD:** [Du et al., 2023](https://arxiv.org/abs/2305.14325)
>   - **SAFE:** [Yun et al., 2026](https://arxiv.org/abs/2510.15346)
>   - **CoRE:** [Zeng et al., 2025](https://arxiv.org/abs/2510.13855)
>   </details>
> - 📊 **Evaluation on multiple benchmarks:** GSM8K, MATH, TriviaQA, and AlpacaEval

> 📢 **Welcome to use! If you find this project helpful, please consider giving it a ⭐ star!**

## 💡 Overview

<div align="center">
    <img src="Images/overview-LLM-PeerReview.png" width="70%" alt="LLM-PeerReview Overview"/>
</div>

<details><summary><strong>Framework of LLM-PeerReview</strong></summary><br>

**LLM-PeerReview** is an unsupervised LLM Ensemble method that selects the most ideal response from multiple LLM-generated candidates for each query, harnessing the collective wisdom of multiple models with diverse strengths. It is built on a novel, peer-review-inspired framework that offers a clear and interpretable mechanism, while remaining fully unsupervised for flexible adaptability and generalization.

The proposed LLM-PeerReview contains three steps: 

(1) Scoring: For a given query, after each LLM independently generates a response (analogous to a submitted academic paper), LLM-PeerReview applies the LLM-as-a-Judge technique (and the proposed flipped-triple scoring trick), treating each model as a reviewer to assign scores to all candidate responses;

(2) Reasoning: LLM-PeerReview then uses a truth inference algorithm——analogous to a senior reviewer——to estimate a final score for each response. (Notably, for the variant LLM-PeerReview-Weighted, the inference algorithm is performed using score information across all queries, allowing the model to learn each LLM’s scoring behavior using global information from the dataset, thereby enabling fine-grained, reliability-aware score aggregation); 

(3) Selecting the best: Finally, for each query, LLM-PeerReview selects the response with the highest final score as the ensemble output—analogous to how a senior reviewer chooses the best paper from a specific submission pool.

</details>


## 1. Environment & Setup

### 1.1 System Requirements

Before using this project, please ensure your development environment meets the following requirements:

- **Operating System**: Linux (Ubuntu 20.04 or higher recommended)
- **Python Version**: Python 3.10.16 or higher
- **Hardware Requirements**:
  - **GPU**: NVIDIA V100 32GB or equivalent GPU for faster inference with large models (CUDA support required)
  - **Memory (RAM)**: Minimum 16GB RAM
  - **Storage**: At least 100GB of SSD storage for model and data processing
  - **Processor**: Intel i7 or higher, or equivalent AMD processor

### 1.2 Create a Virtual Environment

It is recommended to create a virtual environment to manage dependencies for this project. You can create a virtual environment using `venv` or `conda`.

- **Using `venv`**:
  
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

- **Using `conda`**:
  
  ```bash
  conda create -n llm-peerreview python=3.10.16
  conda activate llm-peerreview
  ```

### 1.3 Install Dependencies

Install all required dependencies by running:

```bash
pip install -r requirements.txt
```

If you haven't downloaded the required large models yet, you can use the following script to download them:

```bash
bash ./Script/LLM_Download.sh
```

This will automatically download the necessary models and set them up in the appropriate directories.



## 2.Usage
This project is organized into a structured pipeline for generating, scoring, ensembling, and evaluating LLM responses. 

### Workflow Summary

1. **Generate Responses**: Use scripts in `Script/Response_Generate/` to obtain candidate responses from individual models.
2. **Score Responses**: Employ the PeerReview scoring scripts in `Script/Response_Scoring/` to have LLMs evaluate each other's responses.
3. **Apply Ensemble Methods**: Run scripts in `Script/Ensemble_Generate/` to produce a single best response per query using various selection or aggregation strategies.
4. **Evaluate Results**: Use the scripts in `Script/Response_Evaluation/` to assess and compare the performance of individual models, scoring outcomes, and different ensemble methods.

### 2.0 Quick Start (Unified Pipeline)

To facilitate reproducibility, we provide pre-generated model responses in the `LLM_Response/` directory. This allows you to jump directly to the scoring and ensemble stages without spending hours on local inference.

We provide a unified pipeline script that integrates all stages—from peer-review scoring to ensemble selection and final evaluation—into a single command.

**Usage:**

```bash
bash ./Script/run_baseline.sh --method <METHOD_NAME> --dataset <DATASET>
```

**Examples:**

```bash
# Run LLM-PeerReview-Average on TriviaQA
bash ./Script/run_baseline.sh --method LLM-PeerReview-Average --dataset TriviaQA

# Run Random baseline on GSM8k (short options)
bash ./Script/run_baseline.sh -m Random -d GSM8k

# Run MAD on GSM8k
bash ./Script/run_baseline.sh --method MAD --dataset GSM8k

# Run CoRE-UniTE-RBF on TriviaQA
bash ./Script/run_baseline.sh --method CoRE-UniTE-RBF --dataset TriviaQA

# Show all available options (-h or --help)
bash ./Script/run_baseline.sh --help
```

<details>
<summary><strong>Supported Methods & Datasets</strong></summary>

| Category | Options |
| --- | --- |
| **Baselines** | `Random`, `Smoothie-Global`, `Smoothie-Local`, `Agent-Forest`, `GaC`, `MAD`, `SAFE-GaC-Heur`, `SAFE-UniTE-Heur`, `CoRE-GaC-RBF`, `CoRE-UniTE-RBF` |
| **Model Selection** | `Single-Model`, `Single-Model-Judge` |
| **Proposed (Ours)** | `LLM-PeerReview-Average`, `LLM-PeerReview-Weighted` |
| **Datasets** | `GSM8k`, `MATH`, `TriviaQA`, `AlpacaEval` |

</details>


### 2.1 Response Generation

Generate responses from various LLMs on benchmark datasets:

- **Standard 7B Models**: Generate responses using four different 7B models (Llama-3.1-8B, Mistral-7B, Qwen2-7B, Qwen2.5-7B) on the target datasets.

  ```bash
  bash ./Script/Response_Generate/New_7B_Response_Generate.sh
  ```

- **AlpacaEval Dataset**: Generate responses with the same four models using the AlpacaEval decoding settings.

  ```bash
  bash ./Script/Response_Generate/New_7B_Response_Generate_Alpaca.sh
  ```

**Output**: Generated responses are saved in the `LLM_Response/` directory with organized subfolders for each model and dataset.

Models and datasets can be selected through the `MODELS` and `DATASETS` arrays in the corresponding script.

### 2.2 Response Scoring (PeerReview Method)

Score the generated responses using our PeerReview methodology. We provide scoring scripts for different datasets:

- **GSM8K Dataset**:

  ```bash
  bash ./Script/Response_Scoring/judge/judge_gsm8k400.sh
  ```

- **MATH Dataset**:

  ```bash
  bash ./Script/Response_Scoring/judge/judge_math400.sh
  ```

- **TriviaQA Dataset**:

  ```bash
  bash ./Script/Response_Scoring/judge/judge_trivia_qa.sh
  ```

**Note**: The scoring process leverages the LLM-as-a-Judge paradigm, where each available LLM acts as a reviewer to evaluate and assign scores to all candidate responses, forming the foundation for subsequent ensemble selection. Judge models, judge modes, and score ranges can be selected in the corresponding script.

The unified pipeline sets the dataset-specific scoring options automatically, including the AlpacaEval configuration.

### 2.3 Ensemble Methods

Combine multiple model responses using different ensemble strategies. We compare our proposed method against several established baselines:

- <details><summary><strong>1) Random</strong>: A random-selection baseline that returns a response from a randomly chosen LLM in the ensemble.</summary>

  ```bash
  bash ./Script/Ensemble_Generate/Random_Generate.sh
  ```
  </details>

- <details><summary><strong>2) GaC</strong>: A recent token-level ensemble-during-inference method.</summary>

  ```bash
  bash ./Script/Response_Generate/GaC_7B_Response_Generate.sh
  ```

  GaC response generation runs through the separate server in `GaC/`. Use a separate environment for this service, install its dependencies, and start the server before running this script:

  ```bash
  cd GaC
  pip install -r requirements.txt
  python gac_api_server.py --config-path configs/new_7b_4_model.yaml --host 0.0.0.0 --port 8000
  ```

  The repository also includes pre-generated GaC responses, so the `GaC` option in `run_baseline.sh` evaluates those responses directly.
  </details>

- <details><summary><strong>3) Agent Forest</strong>: A recently proposed similarity-based ensemble method.</summary>

  ```bash
  bash ./Script/Ensemble_Generate/Agent_forest_Generate.sh
  ```
  </details>

- <details><summary><strong>4) Smoothie-Global & Smoothie-Local</strong>: Strong similarity-based ensemble methods that operate at the global level and the local level, respectively.</summary>

  ```bash
  # Global variant
  bash ./Script/Ensemble_Generate/Smoothie-Global_Generate.sh
  
  # Local variant
  bash ./Script/Ensemble_Generate/Smoothie-Local_Generate.sh
  ```
  </details>

- <details><summary><strong>5) Multi-Agent Debate (MAD)</strong>: Models refine their responses using the other agents' responses over multiple debate rounds.</summary>

  ```bash
  bash ./Script/Ensemble_Generate/MAD_Generate.sh
  ```

  The provided script runs GSM8K and MATH and saves Round 0 followed by two debate rounds. The number of debate rounds is controlled by `N_DEBATE_ROUNDS`.
  </details>

- <details><summary><strong>6) SAFE</strong>: A speculative token-level ensemble method that uses a drafter and multiple verifier models.</summary>

  ```bash
  # GaC alignment with heuristic sharpening
  bash ./Script/Ensemble_Generate/SAFE_Generate_GaC_Heur.sh

  # UniTE alignment with heuristic sharpening
  bash ./Script/Ensemble_Generate/SAFE_Generate_UniTE_Heur.sh
  ```

  SAFE profiles are defined in `Configs/safe/`. Datasets and GPU assignments can be changed through `DATASETS` and `DEVICE_IDS` in the generation scripts.
  </details>

- <details><summary><strong>7) CoRE</strong>: A consistency-aware token-level ensemble method with GaC or UniTE token alignment.</summary>

  ```bash
  # GaC alignment with consist-rbf weighting
  bash ./Script/Ensemble_Generate/CoRE_GaC_Generate.sh

  # UniTE alignment with consist-rbf weighting
  bash ./Script/Ensemble_Generate/CoRE_UniTE_Generate.sh
  ```

  Use `CORE_DATASETS_CSV` and `CORE_DEVICES` to select datasets and GPU assignments.
  </details>

- **8) LLM-PeerReview-Average (Ours)**: Our primary ensemble method which averages scores from multiple LLM judges.

  ```bash
  bash ./Script/Ensemble_Generate/PeerReview_Average_Generate.sh
  ```

- **9) LLM-PeerReview-Weighted (Ours)**: An enhanced variant that employs a graphical-model-based truth inference algorithm for reliability-aware score aggregation.

  ```bash
  bash ./Script/Ensemble_Generate/PeerReview_Average_Ti_Generate.sh
  ```

### 2.4 Evaluation

Evaluate the quality of the generated responses and the performance of different ensemble methods:

- <details><summary><strong>1) Single Model Evaluation</strong>: Evaluate responses from the standard 7B models.</summary>

  ```bash
  bash ./Script/Response_Evaluation/New_7B_Response_Evaluate.sh
  ```
  </details>

- <details><summary><strong>2) GaC Baseline Evaluation</strong>: Evaluate responses from the GaC baseline model.</summary>

  ```bash
  bash ./Script/Response_Evaluation/GaC_7B_Response_Evaluate.sh
  ```
  </details>

- <details><summary><strong>3) Scored Response Evaluation</strong>: Evaluate the outcomes after applying the PeerReview scoring process.</summary>

  ```bash
  bash ./Script/Response_Evaluation/New_7B_Judge_Response_Evaluate.sh
  ```
  </details>

- <details><summary><strong>4) Baseline Ensemble Evaluation</strong>: Evaluate the results produced by baseline ensemble methods (Random, Agent Forest, Smoothie).</summary>

  ```bash
  bash ./Script/Response_Evaluation/Baseline_Ensemble_Response_Evaluate.sh
  ```
  </details>

- <details><summary><strong>5) MAD Evaluation</strong>: Evaluate the majority-vote result for each debate round.</summary>

  ```bash
  bash ./Script/Response_Evaluation/MAD_Response_Evaluate.sh
  ```
  </details>

- <details><summary><strong>6) SAFE Evaluation</strong>: Evaluate outputs from the GaC-Heur and UniTE-Heur profiles.</summary>

  ```bash
  bash ./Script/Response_Evaluation/SAFE_Response_Evaluate.sh
  ```
  </details>

- <details><summary><strong>7) CoRE Evaluation</strong>: Evaluate GaC/consist-rbf and UniTE/consist-rbf outputs.</summary>

  ```bash
  bash ./Script/Response_Evaluation/CoRE_Response_Evaluate.sh
  ```
  </details>

- **8) PeerReview Ensemble Evaluation**: Evaluate the final outputs of our proposed PeerReview ensemble methods.

  ```bash
  bash ./Script/Response_Evaluation/PeerReview_Average_Ensemble_Response_Evaluate.sh
  ```

**Note**: The provided GaC, SAFE, and CoRE evaluation scripts do not automatically evaluate AlpacaEval outputs, which require a fresh judge for newly generated text.


## 📚 Citation
```bibtex
@misc{chen2026scoringreasoningselectingbest,
      title={Scoring, Reasoning, and Selecting the Best! Ensembling Large Language Models via a Peer-Review Process},
      author={Zhijun Chen and Zeyu Ji and Qianren Mao and Hao Wu and Jinhuan Song and Junhang Cheng and Bangjie Qin and Zhuoran Li and Jingzheng Li and Kai Sun and Zizhe Wang and Yikun Ban and Zhu Sun and Xiangyang Ji and Hailong Sun and Xiao Huang},
      year={2026},
      eprint={2512.23213},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.23213},
}
```

## License

Project-authored code is released under the MIT License. Third-party components and notices are listed in `THIRD_PARTY_NOTICES.md` and `LICENSES/`.
