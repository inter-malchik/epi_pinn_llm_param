"""config.py — environment variables → LLM_CONFIG.

Everything is computed at import time, so each test reloads the module
through the ``config_env`` fixture (which also disables ``.env`` loading).

Reading guide
-------------
* These are characterization tests: they pin the CURRENT behavior of the code,
  quirks included. A comment starting with ``# NOTE: current behavior`` marks a
  place where the pinned behavior looks like a bug. Fix the code first, then
  update the test on purpose - never "fix" the test to hide the change.
* Test names read as sentences (``test_zero_beta_is_treated_as_missing``); a
  test class groups the tests of one function, method or scenario.
* Only external boundaries are faked (LLM providers, the network). The SIRD
  solver, the PINN, LangGraph and the file system are real. Shared fixtures
  live in ``tests/conftest.py``, fake LLM clients and JSON reply builders in
  ``tests/support.py``.
"""
import importlib

import pytest

LLAMA = "meta-llama/Llama-3.3-70B-Instruct"


def test_defaults_without_env(config_env):
    cfg = config_env()
    assert cfg.LLM_PROVIDER == "huggingface"
    assert cfg.DEFAULT_MODEL_NAME == LLAMA
    assert cfg.DEFAULT_TEMPERATURE == 0.0
    assert cfg.DEFAULT_MAX_TOKENS == 1024
    assert cfg.HUGGINGFACE_TOKEN is None
    assert cfg.HF_MODEL == LLAMA
    assert cfg.HF_USE_API is True
    assert cfg.HF_DEVICE == "cpu"
    assert cfg.OPENAI_MODEL == "gpt-4"
    assert cfg.OPENAI_API_KEY == ""
    assert cfg.VLLM_MODEL == LLAMA
    assert cfg.VLLM_TENSOR_PARALLEL == 1
    assert cfg.LMSTUDIO_BASE_URL == "http://127.0.0.1:1234/v1"
    assert cfg.LMSTUDIO_MODEL == "local-model"


def test_huggingface_is_the_default_branch_of_llm_config(config_env):
    cfg = config_env()
    assert cfg.LLM_CONFIG == {
        "provider": "huggingface",
        "temperature": 0.0,
        "max_tokens": 1024,
        "model_name": LLAMA,
        "kwargs": {"use_api": True, "device": "cpu", "api_token": None},
    }


def test_huggingface_env_overrides(config_env):
    cfg = config_env(
        MODEL_NAME_HF="org/model",
        MODEL_TEMPERATURE_HF="0.5",
        MAX_TOKENS="77",
        HF_USE_API="false",
        HF_DEVICE="cuda",
        HUGGINGFACE_HUB_TOKEN="hf_tok",
    )
    assert cfg.LLM_CONFIG["model_name"] == "org/model"
    assert cfg.LLM_CONFIG["temperature"] == 0.5
    assert cfg.LLM_CONFIG["max_tokens"] == 77
    assert cfg.LLM_CONFIG["kwargs"] == {"use_api": False, "device": "cuda", "api_token": "hf_tok"}
    # the "HF" variables double as the global defaults
    assert cfg.DEFAULT_TEMPERATURE == 0.5
    assert cfg.DEFAULT_MAX_TOKENS == 77


@pytest.mark.parametrize(
    "value,expected",
    [("True", True), ("true", True), ("TRUE", True), ("false", False), ("1", False), ("yes", False), ("", False)],
)
def test_hf_use_api_accepts_only_the_word_true(config_env, value, expected):
    cfg = config_env(HF_USE_API=value)
    assert cfg.HF_USE_API is expected


def test_openai_branch(config_env):
    cfg = config_env(LLM_PROVIDER="openai", OPENAI_MODEL="gpt-x", OPENAI_API_KEY="sk-test")
    assert cfg.LLM_CONFIG == {
        "provider": "openai",
        "temperature": 0.0,
        "max_tokens": 1024,
        "model_name": "gpt-x",
        "kwargs": {"api_key": "sk-test"},
    }


def test_openai_branch_inherits_temperature_from_the_hf_variable(config_env):
    # NOTE: current behavior — DEFAULT_TEMPERATURE/DEFAULT_MAX_TOKENS come from the
    # MODEL_TEMPERATURE_HF / MAX_TOKENS variables even when the provider is OpenAI.
    cfg = config_env(LLM_PROVIDER="openai", MODEL_TEMPERATURE_HF="0.9", MAX_TOKENS="33")
    assert cfg.LLM_CONFIG["temperature"] == 0.9
    assert cfg.LLM_CONFIG["max_tokens"] == 33


def test_vllm_branch(config_env):
    cfg = config_env(
        LLM_PROVIDER="vllm",
        VLLM_MODEL="org/vllm-model",
        VLLM_TEMPERATURE="0.3",
        VLLM_MAX_TOKENS="5",
        VLLM_TENSOR_PARALLEL="2",
    )
    assert cfg.LLM_CONFIG == {
        "provider": "vllm",
        "temperature": 0.3,
        "max_tokens": 5,
        "model_name": "org/vllm-model",
        "kwargs": {"tensor_parallel_size": 2},
    }


def test_vllm_defaults_fall_back_to_hf_temperature_and_tokens(config_env):
    cfg = config_env(LLM_PROVIDER="vllm", MODEL_TEMPERATURE_HF="0.7", MAX_TOKENS="9")
    assert cfg.VLLM_TEMPERATURE == 0.7
    assert cfg.VLLM_MAX_TOKENS == 9
    assert cfg.LLM_CONFIG["temperature"] == 0.7
    assert cfg.LLM_CONFIG["max_tokens"] == 9


def test_vllm_model_default_ignores_model_name_hf(config_env):
    # NOTE: current behavior — VLLM_MODEL has its own literal default, not DEFAULT_MODEL_NAME
    cfg = config_env(LLM_PROVIDER="vllm", MODEL_NAME_HF="org/custom")
    assert cfg.DEFAULT_MODEL_NAME == "org/custom"
    assert cfg.VLLM_MODEL == LLAMA


def test_lmstudio_branch(config_env):
    cfg = config_env(
        LLM_PROVIDER="lmstudio",
        LMSTUDIO_BASE_URL="http://10.0.0.5:1234/v1",
        LMSTUDIO_MODEL="qwen",
        LMSTUDIO_TEMPERATURE="0.2",
        LMSTUDIO_MAX_TOKENS="256",
    )
    assert cfg.LLM_CONFIG == {
        "provider": "lmstudio",
        "temperature": 0.2,
        "max_tokens": 256,
        "model_name": "qwen",
        "kwargs": {"base_url": "http://10.0.0.5:1234/v1"},
    }


def test_lmstudio_defaults(config_env):
    cfg = config_env(LLM_PROVIDER="lmstudio")
    assert cfg.LLM_CONFIG["model_name"] == "local-model"
    assert cfg.LLM_CONFIG["kwargs"] == {"base_url": "http://127.0.0.1:1234/v1"}
    assert cfg.LMSTUDIO_TEMPERATURE == 0.0
    assert cfg.LMSTUDIO_MAX_TOKENS == 1024


def test_unknown_provider_leaves_model_name_unset(config_env):
    # NOTE: current behavior — no branch matches, so LLM_CONFIG has no "model_name";
    # the failure surfaces later in LLMFactory, not here.
    cfg = config_env(LLM_PROVIDER="banana")
    assert cfg.LLM_PROVIDER == "banana"
    assert cfg.LLM_CONFIG == {"provider": "banana", "temperature": 0.0, "max_tokens": 1024, "kwargs": {}}
    assert "model_name" not in cfg.LLM_CONFIG


def test_provider_is_case_sensitive(config_env):
    cfg = config_env(LLM_PROVIDER="OpenAI")
    assert "model_name" not in cfg.LLM_CONFIG


@pytest.mark.parametrize("var", ["MODEL_TEMPERATURE_HF", "MAX_TOKENS", "VLLM_TENSOR_PARALLEL"])
def test_non_numeric_values_break_the_import(config_env, var):
    with pytest.raises(ValueError):
        config_env(**{var: "not-a-number"})


def test_dotenv_is_searched_from_the_current_working_directory(monkeypatch):
    import dotenv

    import config

    calls = {}

    def fake_find(**kw):
        calls["find"] = kw
        return "/nowhere/.env"

    def fake_load(path):
        calls["load"] = path
        return False

    monkeypatch.setattr(dotenv, "find_dotenv", fake_find)
    monkeypatch.setattr(dotenv, "load_dotenv", fake_load)
    try:
        importlib.reload(config)
        assert calls["find"] == {"usecwd": True}
        assert calls["load"] == "/nowhere/.env"
    finally:
        monkeypatch.undo()
        importlib.reload(config)
