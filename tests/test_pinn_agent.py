"""agents/PINNAgent.py — training EINN_PINN from pipeline state.

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
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.PINN_const import EINN_PINN
from agents.PINNAgent import PINNAgent, _ensure_dir, _run_tag
from tests.conftest import ITALY_CSV, REAL_CSV, SYNTHETIC_CSV
from tests.support import make_episode


@pytest.fixture
def agent():
    return PINNAgent(pinn_class=EINN_PINN, n_epoch=2, device="cpu", verbose=False)


# --------------------------------------------------------------------------- helpers & defaults


class TestModuleHelpers:
    """Module-level helpers _run_tag and _ensure_dir."""

    def test_run_tag_format(self):
        assert re.fullmatch(r"iter3_\d{8}_\d{6}", _run_tag(3))

    def test_ensure_dir_creates_and_returns(self, tmp_path):
        target = tmp_path / "a" / "b"
        assert _ensure_dir(str(target)) == str(target)
        assert target.is_dir()
        assert _ensure_dir(str(target)) == str(target)  # idempotent


class TestDefaults:
    """Constructor defaults and device selection of PINNAgent."""

    def test_constructor_defaults(self):
        a = PINNAgent(pinn_class=EINN_PINN)
        assert a.n_epoch == 10_000
        assert (a.lambda_data, a.lambda_ode, a.lambda_ic, a.lambda_bc) == (0.01, 1.0, 0.1, 0.1)
        assert a.results_dir == "PINN_agent_results"
        assert a.verbose is True
        assert a.pinn_class is EINN_PINN

    def test_device_autodetect(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert PINNAgent(pinn_class=EINN_PINN).device == torch.device("cpu")

    def test_explicit_device(self):
        assert PINNAgent(pinn_class=EINN_PINN, device="cpu").device == torch.device("cpu")


# --------------------------------------------------------------------------- _get_params


class TestGetParams:
    """Priority order in which PINNAgent picks β, γ, μ from the pipeline state."""

    def test_last_accepted_episode_wins(self, agent):
        state = {
            "history": [make_episode(iteration=0), make_episode(iteration=1, beta=0.2, accepted=False), make_episode(iteration=2, beta=0.3, accepted=True)],
            "generated_params": {"beta": 0.9, "gamma": 0.1, "mu": 0.01},
            "critic_decision": "accept",
        }
        assert agent._get_params(state) == {"beta": 0.3, "gamma": 0.0553, "mu": 0.0085}

    def test_baseline_counts_as_accepted(self, agent):
        state = {"history": [make_episode(iteration=0)], "generated_params": {"beta": 0.9, "gamma": 0.1, "mu": 0.01}}
        assert agent._get_params(state)["beta"] == 0.091

    def test_generated_params_when_critic_accepted(self, agent):
        gp = {"beta": 0.2, "gamma": 0.1, "mu": 0.01}
        assert agent._get_params({"history": [], "critic_decision": "accept", "generated_params": gp}) is gp

    def test_generated_params_as_last_resort(self, agent):
        gp = {"beta": 0.2, "gamma": 0.1, "mu": 0.01}
        assert agent._get_params({"history": [], "critic_decision": "reject", "generated_params": gp}) is gp

    def test_zero_beta_is_treated_as_missing(self, agent):
        # NOTE: current behavior — possible bug: `if gp.get("beta")` is falsy for β = 0.0
        gp = {"beta": 0.0, "gamma": 0.1, "mu": 0.01}
        assert agent._get_params({"history": [], "critic_decision": "reject", "generated_params": gp}) is None

    def test_zero_beta_is_fine_when_the_critic_accepted(self, agent):
        gp = {"beta": 0.0, "gamma": 0.1, "mu": 0.01}
        assert agent._get_params({"history": [], "critic_decision": "accept", "generated_params": gp}) is gp

    def test_nothing_available(self, agent):
        assert agent._get_params({}) is None
        assert agent._get_params({"history": [], "generated_params": {}}) is None


# --------------------------------------------------------------------------- _get_data


class TestGetData:
    """Data sources of PINNAgent: state dict, task_config dict, CSV file - and what happens without any."""

    def test_pinn_data_in_state(self, agent, pinn_data_small):
        t, S, I, R, D, population, train_size = agent._get_data({"pinn_data": pinn_data_small})
        assert t.tolist() == list(range(40))
        assert S.dtype == float and len(I) == 40
        assert population == pytest.approx(1000.0)
        assert train_size == 30

    def test_pinn_data_in_task_config(self, agent, pinn_data_small):
        out = agent._get_data({"task_config": {"pinn_data": pinn_data_small}})
        assert out[6] == 30

    def test_state_pinn_data_wins_over_task_config(self, agent, pinn_data_small):
        other = dict(pinn_data_small, train_size=7)
        out = agent._get_data({"pinn_data": other, "task_config": {"pinn_data": pinn_data_small}})
        assert out[6] == 7

    def test_unpack_defaults(self, agent):
        data = {"S": [90, 80], "I": [10, 20], "R": [0, 0], "D": [0, 0]}
        t, S, I, R, D, population, train_size = agent._unpack_dict(data)
        assert population == 100.0
        assert train_size == int(2 * 0.75)
        assert t.tolist() == [0.0, 1.0]

    def test_unpack_explicit_population(self, agent):
        data = {"S": [90, 80], "I": [10, 20], "R": [0, 0], "D": [0, 0], "population": 5000}
        assert agent._unpack_dict(data)[5] == 5000.0

    def test_csv_path(self, agent):
        t, S, I, R, D, population, train_size = agent._get_data({"task_config": {"data_path": str(SYNTHETIC_CSV)}})
        assert len(t) == 366
        assert population == pytest.approx(1000.0)
        assert train_size == int(366 * 0.75)

    def test_csv_train_size_from_task_config(self, agent):
        out = agent._get_data({"task_config": {"data_path": str(SYNTHETIC_CSV), "train_size": 120}})
        assert out[6] == 120

    def test_real_csv_column_order_does_not_matter(self, agent):
        _, S, I, R, D, population, _ = agent._load_csv(str(REAL_CSV), {})
        assert population == pytest.approx(4197 + 42 + 5995513 + 248)
        assert I[0] == 4197 and D[0] == 42

    def test_italy_csv_population_is_the_first_row_sum(self, agent):
        out = agent._load_csv(str(ITALY_CSV), {})
        assert out[5] == 60461826.0

    def test_missing_csv_returns_none_and_prints(self, capsys):
        agent = PINNAgent(pinn_class=EINN_PINN, verbose=True)
        assert agent._get_data({"task_config": {"data_path": "nope.csv"}}) is None
        assert "❌ Ошибка загрузки CSV nope.csv" in capsys.readouterr().out

    def test_csv_without_required_columns_returns_none(self, agent, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("S,I\n1,2\n")
        assert agent._get_data({"task_config": {"data_path": str(bad)}}) is None

    def test_no_source_at_all(self, capsys):
        agent = PINNAgent(pinn_class=EINN_PINN, verbose=True)
        assert agent._get_data({"task_config": {"x": 1}, "iteration": 2}) is None
        out = capsys.readouterr().out
        assert "⚠️  Нет pinn_data в state и нет data_path в task_config" in out
        assert "Доступные ключи в state: ['task_config', 'iteration']" in out


# --------------------------------------------------------------------------- _build_model


class TestBuildModel:
    """How PINNAgent seeds the network's latent parameters from the generator's values."""

    def test_latents_are_overwritten_with_the_given_parameters(self, agent, pinn_arrays):
        t, S, I, R, D, population, train_size = pinn_arrays
        model = agent._build_model(0.077, 0.05, 0.004, t, S, I, R, D, population, train_size)
        assert isinstance(model, EINN_PINN)
        assert model.params.get_params_dict() == pytest.approx({"beta": 0.077, "gamma": 0.05, "mu": 0.004}, abs=1e-6)
        assert model.params.beta_latent.item() == pytest.approx(np.arctanh(2 * 0.077 - 1), rel=1e-5)
        assert model.params.beta_latent.dtype == torch.float32

    def test_latents_stay_frozen(self, agent, pinn_arrays):
        # NOTE: current behavior — the block re-enabling gradients is commented out in _build_model
        t, S, I, R, D, population, train_size = pinn_arrays
        model = agent._build_model(0.1, 0.05, 0.004, t, S, I, R, D, population, train_size)
        assert all(p.requires_grad is False for p in model.params.parameters())
        assert len(model.all_params) == 11

    def test_out_of_range_values_are_clipped(self, agent, pinn_arrays):
        t, S, I, R, D, population, train_size = pinn_arrays
        model = agent._build_model(2.5, 0.05, 0.004, t, S, I, R, D, population, train_size)
        assert model.params.beta.item() == pytest.approx(1.0)


# --------------------------------------------------------------------------- train


class TestTrain:
    """PINNAgent.train(): result payload, plot files, error reporting, forwarding of loss weights."""

    def test_successful_training_payload_and_plots(self, agent, pinn_arrays, tmp_path):
        t, S, I, R, D, population, train_size = pinn_arrays
        res = agent.train(0.1, 0.06, 0.003, t, S, I, R, D, population, train_size, tag="run1")
        assert res["success"] is True
        assert res["tag"] == "run1"
        assert res["final_params"] == pytest.approx({"beta": 0.1, "gamma": 0.06, "mu": 0.003}, abs=1e-6)
        assert all(type(v) is float for v in res["final_params"].values())
        assert len(res["losses"]) == 2
        assert [Path(p).name for p in res["plot_paths"]] == ["run1_fit.png", "run1_loss.png", "run1_infected.png"]
        for p in res["plot_paths"]:
            assert Path(p).is_file() and Path(p).stat().st_size > 0
            assert Path(p).resolve().parent == (tmp_path / "PINN_agent_results").resolve()

    def test_custom_results_dir(self, pinn_arrays, tmp_path):
        agent = PINNAgent(pinn_class=EINN_PINN, n_epoch=1, device="cpu", results_dir="out/plots", verbose=False)
        t, S, I, R, D, population, train_size = pinn_arrays
        res = agent.train(0.1, 0.06, 0.003, t, S, I, R, D, population, train_size)
        assert (tmp_path / "out" / "plots" / "run_fit.png").is_file()
        assert res["tag"] == "run"

    def test_zero_epochs_skips_the_loss_plot(self, pinn_arrays):
        agent = PINNAgent(pinn_class=EINN_PINN, n_epoch=0, device="cpu", verbose=False)
        t, S, I, R, D, population, train_size = pinn_arrays
        res = agent.train(0.1, 0.06, 0.003, t, S, I, R, D, population, train_size, tag="z")
        assert res["losses"] == []
        assert [Path(p).name for p in res["plot_paths"]] == ["z_fit.png", "z_infected.png"]

    def test_training_uses_the_agent_loss_weights(self, pinn_arrays, monkeypatch):
        seen = {}
        original = EINN_PINN.train_model

        def spy(self, **kwargs):
            seen.update(kwargs)
            return original(self, **kwargs)

        monkeypatch.setattr(EINN_PINN, "train_model", spy)
        agent = PINNAgent(pinn_class=EINN_PINN, n_epoch=1, lambda_data=2.0, lambda_ode=3.0, lambda_ic=4.0, lambda_bc=5.0, device="cpu", verbose=False)
        t, S, I, R, D, population, train_size = pinn_arrays
        agent.train(0.1, 0.06, 0.003, t, S, I, R, D, population, train_size)
        assert seen == {"n_epoch": 1, "lambda_data": 2.0, "lambda_ode": 3.0, "lambda_ic": 4.0, "lambda_bc": 5.0}

    def test_constructor_failure_is_reported_not_raised(self, pinn_arrays):
        class Broken:
            def __init__(self, **kwargs):
                raise RuntimeError("cannot build")

        agent = PINNAgent(pinn_class=Broken, n_epoch=1, device="cpu", verbose=False)
        t, S, I, R, D, population, train_size = pinn_arrays
        assert agent.train(0.1, 0.06, 0.003, t, S, I, R, D, population, train_size, tag="b") == {
            "success": False,
            "error": "cannot build",
            "tag": "b",
        }


# --------------------------------------------------------------------------- __call__


class TestCall:
    """PINNAgent as a graph node (__call__): guard branches and a full one-epoch run."""

    def test_no_parameters(self, agent):
        state = agent({"history": [], "generated_params": {}})
        assert state["pinn_results"]["success"] is False
        assert state["pinn_results"]["error"].startswith("Нет параметров")

    def test_no_data(self, agent):
        state = agent({"history": [], "critic_decision": "accept", "generated_params": {"beta": 0.1, "gamma": 0.06, "mu": 0.003}, "task_config": {}})
        assert state["pinn_results"]["success"] is False
        assert state["pinn_results"]["error"].startswith("Нет данных")

    def test_full_run_from_state(self, agent, pinn_data_small):
        state = {
            "history": [],
            "critic_decision": "accept",
            "generated_params": {"beta": 0.1, "gamma": 0.06, "mu": 0.003},
            "pinn_data": pinn_data_small,
            "iteration": 4,
        }
        out = agent(state)
        assert out is state
        res = state["pinn_results"]
        assert res["success"] is True
        assert res["tag"].startswith("iter4_")
        assert len(res["plot_paths"]) == 3

    def test_verbose_output(self, pinn_data_small, capsys):
        agent = PINNAgent(pinn_class=EINN_PINN, n_epoch=1, device="cpu", verbose=True)
        agent({"history": [], "critic_decision": "accept", "generated_params": {"beta": 0.1, "gamma": 0.06, "mu": 0.003}, "pinn_data": pinn_data_small})
        out = capsys.readouterr().out
        assert "🧠 PINN AGENT" in out
        assert "📐 Параметры: β=0.1000, γ=0.0600, μ=0.00300" in out
        assert "📊 Использую pinn_data из state" in out
        assert "📊 Данные: 40 точек, train_size=30" in out
        assert "✅ Обучение завершено" in out

    def test_verbose_failure_message(self, pinn_data_small, capsys):
        class Broken:
            def __init__(self, **kwargs):
                raise RuntimeError("cannot build")

        agent = PINNAgent(pinn_class=Broken, n_epoch=1, device="cpu", verbose=True)
        state = agent({"history": [], "critic_decision": "accept", "generated_params": {"beta": 0.1, "gamma": 0.06, "mu": 0.003}, "pinn_data": pinn_data_small})
        assert state["pinn_results"]["success"] is False
        assert "❌ Ошибка PINN: cannot build" in capsys.readouterr().out

    def test_quiet_mode_prints_nothing(self, agent, pinn_data_small, capsys):
        agent({"history": [], "critic_decision": "accept", "generated_params": {"beta": 0.1, "gamma": 0.06, "mu": 0.003}, "pinn_data": pinn_data_small})
        assert "PINN AGENT" not in capsys.readouterr().out


# --------------------------------------------------------------------------- verbose branches


class TestVerboseMessages:
    """Console output of the branches that were exercised quietly above."""

    @pytest.fixture
    def loud(self):
        return PINNAgent(pinn_class=EINN_PINN, n_epoch=1, device="cpu", verbose=True)

    def test_no_parameters_message(self, loud, capsys):
        loud({"history": [], "generated_params": {}})
        assert "❌ Нет параметров для PINN" in capsys.readouterr().out

    def test_no_data_message(self, loud, capsys):
        loud({"history": [], "critic_decision": "accept", "generated_params": {"beta": 0.1, "gamma": 0.06, "mu": 0.003}, "task_config": {}})
        assert "❌ Нет данных для обучения PINN" in capsys.readouterr().out

    def test_task_config_source_message(self, loud, pinn_data_small, capsys):
        loud._get_data({"task_config": {"pinn_data": pinn_data_small}})
        assert "📊 Использую pinn_data из task_config" in capsys.readouterr().out

    def test_csv_source_messages(self, loud, capsys):
        loud._get_data({"task_config": {"data_path": str(SYNTHETIC_CSV), "train_size": 100}})
        out = capsys.readouterr().out
        assert f"📂 Загружаю данные из {SYNTHETIC_CSV}" in out
        assert f"📂 Данные загружены из {SYNTHETIC_CSV}: 366 точек, train_size=100" in out
