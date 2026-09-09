"""Shared fixtures for the characterization test suite.

These tests pin down CURRENT behavior of epi_pinn_llm_param (including
quirks) so the code can be restructured safely. Guarantees provided here:

- every test runs inside its own temporary directory: the agents write
  ``logs/prompts``, ``PINN_agent_results``, ``PINN_verification_*`` and
  ``PINN_comparison_results`` relative to cwd, and the repository must stay
  clean. The project root is on ``sys.path`` so imports work from anywhere;
- torch / numpy seeds are fixed before every test;
- no test can reach the real network (HuggingFace, OpenAI, LM Studio,
  mermaid.ink) — outbound connections raise immediately;
- ``RetryParser`` never sleeps for real (its 1/2/4 s back-off is patched out);
- ``config_env`` reloads ``config`` under a controlled environment without
  reading the developer's ``.env``.
"""
import importlib
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "Phase 1 (model calibration)"
SYNTHETIC_CSV = DATA_DIR / "synthetic_datasets" / "01_baseline_constant.csv"
REAL_CSV = DATA_DIR / "real_datasets" / "covid-19_Kouprianov.csv"
ITALY_CSV = DATA_DIR / "real_datasets" / "PINN-COVID-Italy.csv"

# Every variable config.py reads. `config_env` clears them all first.
CONFIG_ENV_VARS = [
    "LLM_PROVIDER",
    "MODEL_NAME_HF",
    "MODEL_TEMPERATURE_HF",
    "MAX_TOKENS",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_USE_API",
    "HF_DEVICE",
    "OPENAI_MODEL",
    "OPENAI_API_KEY",
    "VLLM_MODEL",
    "VLLM_TEMPERATURE",
    "VLLM_MAX_TOKENS",
    "VLLM_TENSOR_PARALLEL",
    "LMSTUDIO_BASE_URL",
    "LMSTUDIO_MODEL",
    "LMSTUDIO_TEMPERATURE",
    "LMSTUDIO_MAX_TOKENS",
]


# --------------------------------------------------------------------------- autouse guards


@pytest.fixture(autouse=True)
def _run_in_tmp_cwd(tmp_path, monkeypatch):
    """Run every test from its own temp dir so runtime artifacts never land in the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _fixed_seeds():
    torch.manual_seed(0)
    np.random.seed(0)


class _NetworkBlockedError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail fast if a test forgets to mock an external service.

    Local connections are allowed; anything else raises.
    """
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and (
            host in ("127.0.0.1", "::1", "localhost") or host.startswith("/")
        ):
            return real_connect(self, address)
        raise _NetworkBlockedError(
            f"Test attempted a real network connection to {address!r}. "
            "Mock the external service instead."
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """RetryParser backs off 1/2/4 s between retries; never wait for real."""
    import utils.RetryParser as retry_module

    monkeypatch.setattr(retry_module.time, "sleep", lambda *_: None)


# --------------------------------------------------------------------------- config


@pytest.fixture
def config_env(monkeypatch):
    """Reload ``config`` under a controlled environment.

    Usage::

        cfg = config_env(LLM_PROVIDER="openai", OPENAI_API_KEY="sk")

    All config-related variables are removed first and ``.env`` is not read,
    so the result depends only on the values passed. The original module state
    is restored afterwards.
    """
    import dotenv

    import config

    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(dotenv, "find_dotenv", lambda *a, **k: "")

    def reload(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config)

    yield reload

    monkeypatch.undo()
    importlib.reload(config)


# --------------------------------------------------------------------------- LLM doubles


@pytest.fixture
def chat_llm():
    """Factory for a LangChain fake chat model that replays canned replies in order."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    def make(*responses: str):
        return FakeListChatModel(responses=list(responses))

    return make


# --------------------------------------------------------------------------- epidemic data


@pytest.fixture
def initial_conditions():
    """Initial conditions of the synthetic baseline dataset (population 1 000)."""
    return {"population": 1000, "S0": 999, "I0": 1, "R0": 0, "D0": 0}


@pytest.fixture
def simulation_params():
    return {"t_max": 400, "num_points": 1000}


@pytest.fixture(scope="session")
def synthetic_df():
    return pd.read_csv(SYNTHETIC_CSV)


@pytest.fixture(scope="session")
def real_df():
    return pd.read_csv(REAL_CSV)


@pytest.fixture
def pinn_data_small(synthetic_df):
    """First 40 days of the synthetic baseline as the ``pinn_data`` dict the pipeline expects."""
    head = synthetic_df.head(40)
    return {
        "S": head["S"].tolist(),
        "I": head["I"].tolist(),
        "R": head["R"].tolist(),
        "D": head["D"].tolist(),
        "train_size": 30,
    }


@pytest.fixture
def pinn_arrays(synthetic_df):
    """(t, S, I, R, D, population, train_size) for 30 synthetic days — fast PINN construction."""
    head = synthetic_df.head(30)
    S = head["S"].to_numpy(dtype=float)
    I = head["I"].to_numpy(dtype=float)
    R = head["R"].to_numpy(dtype=float)
    D = head["D"].to_numpy(dtype=float)
    t = np.arange(len(S), dtype=float)
    return t, S, I, R, D, float(S[0] + I[0] + R[0] + D[0]), 20


@pytest.fixture
def surrogate_agent():
    from agents.SurrogateModel import SurrogateAgent

    return SurrogateAgent(verbose=False)


@pytest.fixture
def baseline_results(surrogate_agent, initial_conditions, simulation_params):
    """Surrogate run with the synthetic ground-truth parameters (β=0.1, γ=0.06, μ=0.003)."""
    return surrogate_agent.simulate(
        beta=0.1, gamma=0.06, mu=0.003, **initial_conditions, **simulation_params
    )
