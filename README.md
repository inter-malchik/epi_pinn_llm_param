# Expert-Guided Epidemic Forecasting with PINNs and LLMs

<div align="center">

[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-yellow.svg)]()
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

</div>

**Translating qualitative expert knowledge into the mathematical parameters of epidemiological models.**

A framework that combines Physics-Informed Neural Networks (PINNs), Large Language Models (LLMs), and classical SIRD models for controllable, interpretable epidemic forecasting.

---

## Table of Contents

- [Background and Motivation](#background-and-motivation)
- [Key Problem](#key-problem)
- [Features](#features)
- [Demo and Results](#demo-and-results)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
  - [1. Installation](#1-installation)
  - [2. Running the Pipeline](#2-running-the-pipeline)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Background and Motivation

Epidemic forecasting is a high-stakes domain: the cost of error is measured in human lives. Although “black boxes” (deep neural networks) can be accurate, they lack the transparency required for trust. Classical SIR/SEIRD models, by contrast, are transparent and interpretable, but often not accurate enough.

Hybrid approaches such as **Physics-Informed Neural Networks (PINNs)** promise the best of both worlds: neural-network accuracy together with consistency with physical/biological laws. They still miss a critical component — **expert knowledge**.

Epidemiologists constantly revise forecasts using qualitative, informal factors: cultural context, delayed lockdowns, or news about viral variability.

**This project is an attempt to bridge natural-language expert input and the mathematics of the models.**

---

## Key Problem

We identified and empirically confirmed a fundamental issue in hybrid neuro-mechanistic systems (PINNs):

> **Parametric interpretability does not guarantee controllability of model behavior.**

Experiments show that when epidemiological parameters are fixed after being changed according to expert logic, a PINN can produce forecast dynamics **opposite** to what is expected. For example, increasing the transmission rate (`β`) can *lower* the predicted epidemic peak, which violates basic SIRD principles. This finding calls into question the direct use of PINNs in expert-driven systems.

---

## Features

- **End-to-end pipeline:** Automatic chain from a textual expert comment to a validated forecast.
- **Multi-agent LLM system:** Intelligent translation of qualitative requests (“the peak should be lower and later”) into numerical parameter values (`β`, `γ`, `μ`).
- **PINN calibration and validation:** Physics-Informed Neural Networks for parameter estimation and final forecast checks.
- **Built-in sensitivity module:** Deterministic analysis of how each parameter affects key epidemic metrics, so the search for values is fast and justified.
- **Reproducible testing:** Experiments on synthetic data and real COVID-19 data (St. Petersburg).

---

## Demo and Results

The key observation is a systematic mismatch between the surrogate SIRD forecast and the final PINN forecast under the same parameters.

**LLM-controlled experiment (“The peak should be lower”):**

Parameter `β` was decreased, which for the classical model means a lower and later peak. The PINN with the same parameters, however, predicted an **increase** in the peak.

---

## How It Works

The framework is built on `LangGraph` and consists of three phases, organized as a directed workflow graph:

![Pipeline diagram](pipeline_graph.png)

### Phase 1: PINN Calibration and Initial Baseline

A Physics-Informed Neural Network is trained on historical epidemiological data. It solves the inverse problem: recovering compartment trajectories (S, I, R, D) and identifying baseline parameters `β, γ, μ`.

### Phase 2: LLM-Guided Parameter Optimization (LangGraph Pipeline)

This is the core of the system, split into several nodes:

1. **`Intent Parser` (LLM):** Parses the expert request (“I want a higher peak”) and formalizes it as a target action (`peak_height: increase`).
2. **`Sensitivity Node` (Deterministic):** Runs a fast sensitivity analysis on the surrogate SIRD model to see how changes in `β, γ, μ` affect the peak.
3. **`LLM Parameter Generator` (LLM):** Given the goal, sensitivity map, and attempt history, generates a new adjusted parameter set.
4. **`Surrogate Evaluator + Deterministic Critic`:** Quickly checks the generated parameters on the classical SIRD model. If the expected effect is achieved, the parameters proceed; otherwise the loop repeats.

### Phase 3: Final Verification with Frozen Parameters

Optimized parameters are frozen and fed back into the PINN to obtain the final forecast and compare it with the baseline. This is the stage where the key mismatch appears.

---

## Quick Start

### 1. Installation

Clone the repository and install dependencies.

```bash
# Clone the repository
git clone https://github.com/vnlenenko/Epi_PINN_LLM_param.git
cd Epi_PINN_LLM_param

# Recommended: create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file with LLM provider settings (if needed).

### 2. Running the Pipeline

```bash
python main_test.py
```

The provider is set by `LLM_PROVIDER` in `.env` (default `huggingface`; also `openai`, `vllm`, `lmstudio`).

---

## Project Structure

```
epi_pinn_llm_param/
├── main_test.py                 # Entry point: LangGraph pipeline (Phases 2–3) + PINN comparison
├── config.py                    # LLM provider settings (from .env)
├── .env                         # Secrets and provider choice (not committed)
├── requirements.txt
├── expert_comment_peak_examples.txt
├── pipeline_graph.png           # Exported LangGraph diagram
├── Makefile                     # install / test / test-fast / coverage / lint / run
├── pytest.ini                   # test-runner configuration
├── requirements-dev.txt         # pytest, pytest-cov, pytest-mock, pylint
├── .github/workflows/tests.yml  # CI: pytest + coverage on every push/PR, pylint report
│
├── tests/                       # Characterization test suite (see Testing)
│   ├── conftest.py              # Shared fixtures: temp cwd, seeds, network block, config reload
│   ├── support.py               # Fake LLM clients and JSON reply builders
│   └── test_*.py                # One file per module / layer
│
├── agents/                      # Runtime components used by the graph
│   ├── BaseLLMClient.py         # Abstract client + LLMResponse
│   ├── LLMClients.py            # OpenAI, HuggingFace, vLLM, LM Studio
│   ├── LLMFactory.py            # Builds a client from config.LLM_CONFIG
│   ├── IntentParserAgent.py     # Expert comment → ExpertIntent (peak direction)
│   ├── EpiParamGeneratorAgent.py# New β, γ, μ + reasoning (EpiParameters)
│   ├── DeterministicCriticAgent.py  # Default critic: accept/reject vs intent
│   ├── DeterministicCriticAgent2.py # Experimental critic variants (not in default graph)
│   ├── DeterministicCriticAgent3.py
│   ├── ParameterCriticAgent.py  # Optional LLM-only critic (not in default graph)
│   ├── SurrogateModel.py        # Classical SIRD (scipy ODE) + SurrogateAgent
│   ├── PINN_const.py            # EINN_PINN network and frozen EpiParams
│   └── PINNAgent.py             # Train PINN from pipeline state
│
├── formats/
│   └── data_formats.py          # Pydantic I/O schemas and PipelineState
│
├── utils/
│   ├── PromptLogger.py          # Saves generator/critic prompts under logs/prompts/
│   └── RetryParser.py           # Retries Pydantic parsing of LLM JSON
│
├── Phase 1 (model calibration)/ # Offline calibration → baseline β, γ, μ
│   ├── PINN_test.ipynb
│   ├── SIRD_calibration.ipynb
│   ├── synthetic_datasets/
│   └── real_datasets/
│
└── PINN_comparison_results/     # Baseline vs optimized PINN plots/JSON
```

Runtime folders created on a run: `PINN_agent_results/` (PINN plots per iteration), `logs/prompts/` (LLM traces).

### Layers

| Layer | Role |
|---|---|
| **Phase 1 notebooks** | Fit baseline `β, γ, μ` on historical or synthetic SIRD data (classical ODE or PINN). |
| **`main_test.py`** | Online pipeline: LangGraph nodes, `OptimizationPipeline`, PINN vs SIRD comparison, reports. |
| **`agents/`** | LLM workers, SIRD surrogate, PINN training. |
| **`formats/`** | Shared contracts so every node reads/writes the same state. |
| **`utils/`** | Prompt logging and robust JSON parsing. |
| **`config.py`** | Maps `.env` to `LLM_CONFIG` (`huggingface` / `openai` / `vllm` / `lmstudio`). |

### LangGraph nodes (`main_test.py`)

```
sensitivity → intent → generate → surrogate → critic → history
                                              ↓ accept
                                    pinn_verification → END
                                              ↓ reject
                                         generate (next iteration)
```

| Node | Kind | What it does |
|---|---|---|
| `sensitivity` | deterministic | Perturbs `β, γ, μ` on SIRD and builds a sensitivity map (peak day / height). |
| `intent` | LLM | Parses the expert comment into `ExpertIntent` (higher/lower, earlier/later). |
| `generate` | LLM | Proposes new `β, γ, μ` from intent, sensitivity, and rejected attempts. |
| `surrogate` | deterministic | Integrates SIRD; records peak position, peak height, deaths. |
| `critic` | mixed | Accepts the episode if the surrogate peak moved in the requested direction. |
| `history` | deterministic | Appends an `Episode`; loops to `generate` or stops. |
| `pinn_verification` | PINN | Retrains `EINN_PINN` with **frozen** accepted parameters and compares to baseline. |

Example comments (peak only, combinations, interventions) are in `expert_comment_peak_examples.txt`.

### Shared schemas (`formats/data_formats.py`)

- **`EpiParameters`** — LLM generator output: `beta`, `gamma`, `mu`, `reasoning`, `confidence`.
- **`ExpertIntent`** — whether the expert cares about peak **position** and/or **height**, and the direction.
- **`Episode`** — one iteration: parameters, peak metrics, expert comment, `accepted` flag.
- **`PipelineState`** — LangGraph state: task config, history, generated params, surrogate/PINN results, iteration counters.

### Data

Phase 1 notebooks produce the baseline `β, γ, μ` that `main_test.py` starts from. Switch dataset and comment inside `main()`.

**Synthetic CSV** (`synthetic_datasets/`): `day, S, I, R, D, beta, gamma, mu, R0`

| File | Scenario |
|---|---|
| `01_baseline_constant` | Constant parameters |
| `02_lockdown_beta_jump` | β drop (lockdown); noisy copies at 3/5/10% |
| `03_seasonal_beta_sin` | Seasonal β |
| `04_decaying_beta_trend` | Decaying β |
| `05_complex_beta_gamma_dynamics` | Joint β and γ dynamics; noisy copies |

**Real CSV** (`real_datasets/`): typically `t, I, D, S, R`

- `covid-19_Kouprianov.csv` — COVID-19, St. Petersburg
- `PINN-COVID-Italy.csv` — COVID-19, Italy

### Runtime artifacts

| Path | Contents |
|---|---|
| `PINN_comparison_results/` | Baseline vs optimized PINN (`comparison_*.png/json`, `peak_analysis_*.png`, `pdf_plots/`) |
| `PINN_agent_results/` | Per-run PINN training plots |
| `logs/prompts/generator/` | Generator prompts and raw LLM replies |
| `logs/prompts/critic/` | Critic prompts and replies |
| `pipeline_graph.png` | Mermaid export of the compiled graph |

### Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `huggingface` (default), `openai`, `vllm`, or `lmstudio` |
| `MODEL_NAME_HF`, `MODEL_TEMPERATURE_HF`, `MAX_TOKENS` | HuggingFace model |
| `HF_USE_API`, `HF_DEVICE`, `HUGGINGFACE_HUB_TOKEN` | API vs local, device, token |
| `OPENAI_MODEL`, `OPENAI_API_KEY` | OpenAI |
| `VLLM_MODEL`, `VLLM_TENSOR_PARALLEL` | Local vLLM |
| `LMSTUDIO_BASE_URL`, `LMSTUDIO_MODEL` | LM Studio (`http://127.0.0.1:1234/v1`) |

---

## Testing

The repository ships a **characterization test suite**: it pins the *current* behavior of every module (quirks included) so the code can be restructured safely. A test failing after a refactor means observable behavior changed. Places where the pinned behavior looks like a bug are marked `# NOTE: current behavior — possible bug: ...` in the tests — fix the code first, then update the test deliberately.

```bash
make install     # once: creates venv, installs CPU torch + requirements + dev tools
make test        # whole suite (~40 s on CPU)
make test-fast   # without the slow end-to-end runs of main():  -m "not slow"
make coverage    # HTML line-by-line coverage report in htmlcov/
make lint        # pylint, errors and fatals only
```

Or directly: `./venv/bin/python -m pytest`.

| Test file | Covers |
|---|---|
| `test_config.py` | `.env` → `LLM_CONFIG` mapping for every provider |
| `test_data_formats.py` | Pydantic contracts, `Episode`, declared vs. actually used `PipelineState` keys |
| `test_llm_clients.py` | Four LLM clients (SDKs stubbed) and `LLMFactory` |
| `test_utils.py` | `RetryParser` back-off, `PromptLogger` files and metadata |
| `test_surrogate.py` | SIRD integration properties and `SurrogateAgent` as a node |
| `test_pinn.py`, `test_pinn_agent.py` | `EpiParams`, scaler, `EINN_PINN` training / MC Dropout, `PINNAgent` |
| `test_agents_intent_generator.py`, `test_critics.py` | Intent parser, parameter generator, all critic variants |
| `test_pipeline_nodes.py`, `test_pipeline_graph.py` | Every LangGraph node in isolation; graph topology; end-to-end `run()` with a scripted LLM |
| `test_reports.py` | PINN comparison, summary report, comparison plot |
| `test_project_contracts.py` | Dataset schemas, notebook import paths, `main()` constants, dependency pins |
| `test_main_entry.py` (`slow`) | `main()` end to end with training shortened to one epoch |

Design rules: only external boundaries are faked (LLM providers, network — outbound connections raise in `conftest.py`); the ODE solver, PINN, LangGraph and file system are real; every test runs in its own temporary directory so no artifact lands in the repo; seeds are fixed.

CI (`.github/workflows/tests.yml`) runs the full suite with coverage on every push and pull request (Python 3.13, CPU build of torch — `requirements.txt` pins a CUDA wheel that only exists on the PyTorch index) and a non-blocking pylint report.

---

## Citation

If you use this code or framework, please cite:

```bibtex
@inproceedings{gindullina2026interpretable,
  title={Interpretable Expert-Informed Epidemic Forecasting via Hybrid Mechanistic and LLM-Based Modeling},
  author={Gindullina, Dinara and Leonenko, Vasiliy},
  booktitle={Proceedings of ...},
  year={2026}
}
```
