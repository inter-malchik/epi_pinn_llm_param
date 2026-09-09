"""main_test.py — the LangGraph node classes, exercised one by one outside the graph.

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
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import agents.PINN_const as pinn_module
from agents.PINN_const import EINN_PINN
from agents.PINNAgent import PINNAgent
from agents.SurrogateModel import SurrogateAgent
from formats.data_formats import Episode
from main_test import DecisionNode, HistoryNode, PINNNode, PINNVerificationNode, SensitivityNode, SurrogateNode
from tests.support import make_episode

# --------------------------------------------------------------------------- SensitivityNode

BASE = {"beta": 0.1, "gamma": 0.06, "mu": 0.003}


@pytest.fixture
def sens_state(initial_conditions):
    return {"task_config": {**BASE, **initial_conditions}, "simulation_params": {"t_max": 400, "num_points": 500}}


@pytest.fixture
def sens_map(surrogate_agent, sens_state):
    return SensitivityNode(surrogate_agent)(sens_state)["sensitivity_map"]


class TestSensitivityNode:
    """SensitivityNode: map structure, single and combined perturbations, console interpretation (including the x2 scaling)."""

    def test_structure(self, sens_map):
        assert set(sens_map) == {"beta", "gamma", "mu", "combined", "baseline"}
        assert set(sens_map["baseline"]) == {"beta", "gamma", "mu", "peak_position", "peak_height"}
        for key in ("beta", "gamma", "mu"):
            assert sens_map[key]["variations"] == [-0.02, -0.01, 0.01, 0.02]
            assert len(sens_map[key]["results"]) == 4
            for res in sens_map[key]["results"]:
                assert set(res) == {"beta", "gamma", "mu", "peak_position", "peak_height", "position_delta", "height_delta"}

    def test_baseline_matches_a_direct_surrogate_run(self, sens_map, baseline_results):
        # baseline_results uses 1000 points, the node here 500 → compare loosely
        assert sens_map["baseline"]["peak_position"] == pytest.approx(baseline_results["peak_position"], abs=1.0)
        assert sens_map["baseline"]["peak_height"] == pytest.approx(baseline_results["peak_height"], rel=1e-3)
        assert sens_map["baseline"]["beta"] == 0.1

    def test_single_variations_perturb_one_parameter_at_a_time(self, sens_map):
        first_beta = sens_map["beta"]["results"][0]
        assert first_beta["beta"] == pytest.approx(0.1 * 0.98)
        assert first_beta["gamma"] == 0.06 and first_beta["mu"] == 0.003
        last_gamma = sens_map["gamma"]["results"][-1]
        assert last_gamma["gamma"] == pytest.approx(0.06 * 1.02) and last_gamma["beta"] == 0.1
        assert sens_map["mu"]["results"][1]["mu"] == pytest.approx(0.003 * 0.99)

    def test_deltas_are_relative_to_the_baseline(self, sens_map):
        base = sens_map["baseline"]
        for key in ("beta", "gamma", "mu", "combined"):
            for res in sens_map[key]["results"]:
                assert res["position_delta"] == pytest.approx(res["peak_position"] - base["peak_position"])
                assert res["height_delta"] == pytest.approx(res["peak_height"] - base["peak_height"])

    def test_epidemiological_directions(self, sens_map):
        # more transmission → earlier and higher; more recovery → later and lower
        assert sens_map["beta"]["results"][-1]["position_delta"] < 0
        assert sens_map["beta"]["results"][-1]["height_delta"] > 0
        assert sens_map["gamma"]["results"][-1]["position_delta"] > 0
        assert sens_map["gamma"]["results"][-1]["height_delta"] < 0
        assert abs(sens_map["mu"]["results"][-1]["position_delta"]) < 2.0

    def test_combined_variations(self, sens_map):
        assert sens_map["combined"]["variations"] == ["β↑ γ↓", "β↓ γ↑", "β↑ γ↑", "β↓ γ↓", "β↑↑ γ↓↓", "β↓↓ γ↑↑", "β↑ μ↑", "β↓ μ↓"]
        res = sens_map["combined"]["results"]
        assert res[0]["beta"] == pytest.approx(0.105) and res[0]["gamma"] == pytest.approx(0.057) and res[0]["mu"] == 0.003
        assert res[4]["beta"] == pytest.approx(0.11) and res[4]["gamma"] == pytest.approx(0.054)
        assert res[6]["mu"] == pytest.approx(0.003 * 1.05)

    def test_mu_zero_uses_a_synthetic_baseline_for_mu(self, surrogate_agent, initial_conditions):
        state = {"task_config": {"beta": 0.1, "gamma": 0.06, "mu": 0.0, **initial_conditions}, "simulation_params": {"t_max": 200, "num_points": 200}}
        m = SensitivityNode(surrogate_agent)(state)["sensitivity_map"]
        assert [r["mu"] for r in m["mu"]["results"]] == pytest.approx([0.00098, 0.00099, 0.00101, 0.00102])
        assert m["combined"]["results"][6]["mu"] == 0.001

    def test_defaults_when_task_config_is_empty(self, surrogate_agent):
        m = SensitivityNode(surrogate_agent)({})["sensitivity_map"]
        assert m["baseline"] == pytest.approx({"beta": 0.1, "gamma": 0.05, "mu": 0.001, "peak_position": m["baseline"]["peak_position"], "peak_height": m["baseline"]["peak_height"]})
        assert m["beta"]["results"][0]["gamma"] == 0.05

    def test_returns_the_same_state_object(self, surrogate_agent, sens_state):
        out = SensitivityNode(surrogate_agent)(sens_state)
        assert out is sens_state and "sensitivity_map" in sens_state

    def test_console_claims_six_variations_and_scales_deltas_by_two(self, surrogate_agent, sens_state, capsys):
        # NOTE: current behavior — the log says "6 variations" for a 4-element list, and the
        # "+10%" rule of thumb is the ±1/±2 % average multiplied by 2
        m = SensitivityNode(surrogate_agent)(sens_state)["sensitivity_map"]
        out = capsys.readouterr().out
        assert "Computing beta sensitivity (6 variations)" in out
        beta_pos = np.mean([r["position_delta"] for r in m["beta"]["results"]])
        beta_h = np.mean([r["height_delta"] for r in m["beta"]["results"]])
        expected = f"• β +10% → peak moves {'EARLIER' if beta_pos < 0 else 'LATER'} by {abs(beta_pos * 2):.1f} days, height {'INCREASES' if beta_h > 0 else 'DECREASES'} by {abs(beta_h * 2):.0f}"
        assert expected in out
        assert "• μ has minimal effect on peak (< 1 day), mainly affects total deaths" in out
        assert "✅ Extended sensitivity map computed and saved to state" in out

    def test_simulate_helper_delegates_to_the_surrogate(self, surrogate_agent, initial_conditions):
        node = SensitivityNode(surrogate_agent)
        res = node._simulate(0.1, 0.06, 0.003, **initial_conditions, t_max=100, num_points=50)
        assert res["success"] is True and len(res["t"]) == 50

    def test_failed_baseline_simulation_is_a_key_error(self, surrogate_agent, initial_conditions):
        # NOTE: current behavior — a failed run returns {'success': False, ...}, which is truthy,
        # so the node indexes 'peak_position' on it
        state = {"task_config": {**BASE, **initial_conditions}, "simulation_params": {"t_max": 100, "num_points": 0}}
        with pytest.raises(KeyError):
            SensitivityNode(surrogate_agent)(state)


# --------------------------------------------------------------------------- SurrogateNode


class TestSurrogateNode:
    """SurrogateNode wrapper: default initial conditions and console output."""

    def test_no_params(self, surrogate_agent, capsys):
        state = {"generated_params": {}}
        out = SurrogateNode(surrogate_agent)(state)
        assert out is state and "surrogate_results" not in state
        assert "❌ No parameters to evaluate" in capsys.readouterr().out

    def test_initial_conditions_default_from_task_config(self, surrogate_agent):
        state = {"generated_params": BASE, "task_config": {"population": 500, "S0": 490, "I0": 10, "R0": 0, "D0": 0}}
        SurrogateNode(surrogate_agent)(state)
        assert state["initial_conditions"] == {"population": 500, "S0": 490, "I0": 10, "R0": 0, "D0": 0}
        assert state["surrogate_results"]["success"] is True

    def test_initial_conditions_hard_defaults(self, surrogate_agent):
        state = {"generated_params": BASE}
        SurrogateNode(surrogate_agent)(state)
        assert state["initial_conditions"] == {"population": 10_000, "S0": 9_999, "I0": 1, "R0": 0, "D0": 0}

    def test_existing_initial_conditions_are_kept(self, surrogate_agent, initial_conditions):
        state = {"generated_params": BASE, "initial_conditions": initial_conditions, "task_config": {"population": 5}}
        SurrogateNode(surrogate_agent)(state)
        assert state["initial_conditions"] is initial_conditions

    def test_console_success_and_failure(self, surrogate_agent, initial_conditions, capsys):
        SurrogateNode(surrogate_agent)({"generated_params": BASE, "initial_conditions": initial_conditions})
        out = capsys.readouterr().out
        assert "📊 SURROGATE MODEL" in out and re.search(r"✅ Peak: \d+\.\d days", out) and "✅ Deaths:" in out
        SurrogateNode(surrogate_agent)({"generated_params": BASE, "initial_conditions": initial_conditions, "simulation_params": {"t_max": 10, "num_points": 0}})
        assert "❌ Simulation failed: Симуляция не удалась" in capsys.readouterr().out


# --------------------------------------------------------------------------- HistoryNode


class _Critic:
    """Stand-in for the critic: only the history list and add_to_history() are needed by HistoryNode."""
    def __init__(self):
        self.history = []

    def add_to_history(self, ep):
        self.history.append(ep)


class _Generator:
    """Stand-in for the generator: HistoryNode only assigns its .history attribute."""
    def __init__(self):
        self.history = []


def _history_state(**overrides):
    state = {
        "iteration": 2,
        "generated_params": {"beta": 0.095, "gamma": 0.0553, "mu": 0.0085},
        "surrogate_results": {"peak_position": 125.0, "peak_height": 11000.0, "total_deaths": 550.0},
        "critic_decision": "accept",
        "critic_reasoning": "fine",
        "expert_comment": "Need higher peak",
        "history": [make_episode()],
    }
    state.update(overrides)
    return state


class TestHistoryNode:
    """HistoryNode: episode assembly, history synchronisation between critic and generator, failure without a critic."""

    def test_episode_is_built_and_appended(self):
        critic, gen = _Critic(), _Generator()
        state = _history_state()
        original_list = state["history"]
        out = HistoryNode(generator=gen, critic=critic)(state)
        assert out is state
        assert state["history"] is original_list and len(state["history"]) == 2
        ep = state["history"][-1]
        assert isinstance(ep, Episode)
        assert (ep.beta, ep.gamma, ep.mu) == (0.095, 0.0553, 0.0085)
        assert (ep.peak_position, ep.peak_height, ep.total_deaths) == (125.0, 11000.0, 550.0)
        assert ep.iteration == 2 and ep.accepted is True and ep.reasoning == "fine"
        assert ep.expert_comment == "Need higher peak"
        assert state["current_episode"] is ep

    def test_critic_and_generator_histories_are_synchronised_by_reference(self):
        critic, gen = _Critic(), _Generator()
        HistoryNode(generator=gen, critic=critic)(_history_state())
        assert len(critic.history) == 1
        assert gen.history is critic.history

    def test_duplicate_iteration_is_not_added_to_the_critic_twice(self):
        critic = _Critic()
        critic.history.append(make_episode(iteration=2))
        HistoryNode(critic=critic)(_history_state())
        assert len(critic.history) == 1

    def test_rejected_episode(self, capsys):
        state = HistoryNode()(_history_state(critic_decision="reject"))
        assert state["history"][-1].accepted is False
        out = capsys.readouterr().out
        assert "❌ Episode 2 REJECTED and added to history" in out
        assert "📊 Total history: 2 episodes" in out and "- Accepted: 1" in out and "- Rejected: 1" in out

    def test_missing_decision_defaults_to_reject(self):
        state = _history_state()
        del state["critic_decision"]
        assert HistoryNode()(state)["history"][-1].accepted is False

    def test_missing_params_and_results_become_zeros(self):
        state = HistoryNode()(_history_state(generated_params={}, surrogate_results={}))
        ep = state["history"][-1]
        assert (ep.beta, ep.gamma, ep.mu, ep.peak_position, ep.peak_height, ep.total_deaths) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_generator_without_critic_crashes(self):
        # NOTE: current behavior — possible bug: `self.generator.history = self.critic.history`
        # dereferences a None critic
        with pytest.raises(AttributeError):
            HistoryNode(generator=_Generator(), critic=None)(_history_state())

    def test_no_agents_at_all_is_fine(self):
        assert len(HistoryNode()(_history_state())["history"]) == 2

    def test_history_starts_empty_when_absent(self):
        state = _history_state()
        del state["history"]
        assert len(HistoryNode()(state)["history"]) == 1


# --------------------------------------------------------------------------- DecisionNode


class TestDecisionNode:
    """DecisionNode (used only when use_pinn=False): end/continue routing."""

    def test_accept_ends(self):
        state = {"critic_decision": "accept", "iteration": 1, "max_iterations": 5}
        assert DecisionNode()(state) == "end" and state["iteration"] == 1

    def test_max_iterations_ends(self):
        state = {"critic_decision": "reject", "iteration": 5, "max_iterations": 5}
        assert DecisionNode()(state) == "end"

    def test_continue_increments(self):
        state = {"critic_decision": "reject", "iteration": 1, "max_iterations": 5}
        assert DecisionNode()(state) == "continue" and state["iteration"] == 2

    def test_defaults(self):
        assert DecisionNode()({}) == "continue"
        assert DecisionNode()({"iteration": 10}) == "end"


# --------------------------------------------------------------------------- PINNNode


class _FakeAgent:
    """Stand-in for PINNAgent: records the n_epoch value in force at call time."""
    def __init__(self, n_epoch=7):
        self.n_epoch = n_epoch
        self.seen = []

    def __call__(self, state):
        self.seen.append(self.n_epoch)
        state["pinn_results"] = {"success": True}
        return state


class TestPINNNode:
    """PINNNode: one-epoch validation run on accept, skip on reject."""

    def test_accept_runs_one_epoch_then_restores(self):
        agent = _FakeAgent(n_epoch=7)
        node = PINNNode(agent)
        state = node({"critic_decision": "accept"})
        assert agent.seen == [1]
        assert agent.n_epoch == 7
        assert state["pinn_results"] == {"success": True}

    def test_reject_skips(self):
        agent = _FakeAgent()
        state = PINNNode(agent)({"critic_decision": "reject"})
        assert agent.seen == []
        assert state["pinn_results"] == {"success": False, "skipped": True, "reason": "Parameters rejected by critic (decision: reject)"}

    def test_missing_decision_is_reject(self):
        assert PINNNode(_FakeAgent())({})["pinn_results"]["skipped"] is True

    def test_with_the_real_agent(self, pinn_data_small):
        agent = PINNAgent(pinn_class=EINN_PINN, n_epoch=50, device="cpu", verbose=False)
        state = {"critic_decision": "accept", "history": [], "generated_params": BASE, "pinn_data": pinn_data_small}
        PINNNode(agent)(state)
        assert state["pinn_results"]["success"] is True
        assert len(state["pinn_results"]["losses"]) == 1  # trained for exactly one epoch
        assert agent.n_epoch == 50


# --------------------------------------------------------------------------- PINNVerificationNode


@pytest.fixture
def verification_state(surrogate_agent, initial_conditions, pinn_data_small):
    surrogate = surrogate_agent.simulate(**BASE, **initial_conditions, t_max=100, num_points=50)
    return {
        "generated_params": dict(BASE),
        "surrogate_results": surrogate,
        "task_config": {"pinn_data": pinn_data_small},
        "iteration": 3,
    }


class TestPINNVerificationNodeGuards:
    """PINNVerificationNode: early exits and the NameError of the fallback branches."""

    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        node = PINNVerificationNode()
        assert node.n_epoch == 5000 and node.save_plots is True and node.device == "cpu"

    def test_no_params(self):
        state = {"generated_params": {}}
        out = PINNVerificationNode(n_epoch=1)(state)
        assert out is state and state["pinn_verification"] == {"success": False, "error": "No parameters"}

    def test_no_surrogate_results(self):
        state = {"generated_params": BASE, "surrogate_results": {"success": False}}
        assert PINNVerificationNode(n_epoch=1)(state)["pinn_verification"] == {"success": False, "error": "No surrogate data"}

    def test_no_trajectories(self):
        state = {"generated_params": BASE, "surrogate_results": {"success": True, "t": [0, 1]}}
        assert PINNVerificationNode(n_epoch=1)(state)["pinn_verification"] == {"success": False, "error": "No trajectories"}

    @pytest.mark.parametrize("task_config", [{}, {"pinn_data": {"S": [], "I": [], "R": [], "D": []}}])
    def test_fallback_initial_conditions_crash_on_train_size(self, verification_state, task_config):
        # NOTE: current behavior — possible bug: `train_size` is only assigned when real data is
        # present; both fallback branches then hit `print(f'train_size111 = {train_size}')`
        verification_state["task_config"] = task_config
        with pytest.raises(NameError):
            PINNVerificationNode(n_epoch=1, save_plots=False)(verification_state)


class TestPINNVerificationNodeRun:
    """PINNVerificationNode full run: report contents, files on disk, training configuration, MC-Dropout arguments."""

    def test_full_run_writes_report_plots_and_state(self, verification_state, tmp_path, capsys):
        node = PINNVerificationNode(n_epoch=3, save_plots=True)
        out = node(verification_state)
        v = out["pinn_verification"]
        assert v["success"] is True
        assert v["llm_params"] == BASE
        assert v["pinn_recovered_params"] == pytest.approx(BASE, abs=1e-6)  # frozen → unchanged
        assert v["parameter_error"] == pytest.approx({"beta": 0.0, "gamma": 0.0, "mu": 0.0}, abs=1e-6)
        assert v["relative_error"] == pytest.approx({"beta": 0.0, "gamma": 0.0, "mu": 0.0}, abs=1e-5)
        assert len(v["predictions"]["t"]) == 50 and set(v["predictions"]) == {"t", "I", "S", "R", "D"}
        assert v["uncertainty"]["n_passes"] == 100 and len(v["uncertainty"]["ci_lower_95_I"]) == 50
        assert set(v["peak_analysis"]) == {"synthetic", "pinn_recovered", "error"}
        assert v["peak_analysis"]["synthetic"]["day"] == pytest.approx(verification_state["surrogate_results"]["peak_position"], abs=1e-6)
        assert v["train_size"] == 30
        assert isinstance(v["final_loss"], float)
        plots = [Path(p) for p in v["plot_paths"]]
        assert [p.parent.name for p in plots] == ["PINN_verification_plots", "PINN_verification_plots"]
        assert plots[0].name.startswith("verification_") and plots[0].name.endswith("_iter3.png")
        assert plots[1].name.startswith("parameters_") and all(p.is_file() for p in plots)
        json_files = list((tmp_path / "PINN_verification_results").glob("verification_*.json"))
        assert len(json_files) == 1
        assert json.loads(json_files[0].read_text())["peak_analysis"] == v["peak_analysis"]
        console = capsys.readouterr().out
        assert "train_size111 = 30" in console
        assert "Real data initial conditions (from pinn_data)" in console

    def test_no_plots_when_disabled(self, verification_state, tmp_path):
        v = PINNVerificationNode(n_epoch=1, save_plots=False)(verification_state)["pinn_verification"]
        assert v["plot_paths"] == []
        assert not (tmp_path / "PINN_verification_plots").exists()
        assert (tmp_path / "PINN_verification_results").is_dir()

    def test_model_is_trained_on_the_whole_synthetic_trajectory_with_fixed_weights(self, verification_state, monkeypatch):
        seen = {}
        original_init, original_train = EINN_PINN.__init__, EINN_PINN.train_model

        def spy_init(self, *args, **kwargs):
            seen["init"] = kwargs
            return original_init(self, *args, **kwargs)

        def spy_train(self, **kwargs):
            seen["train"] = kwargs
            return original_train(self, **kwargs)

        monkeypatch.setattr(EINN_PINN, "__init__", spy_init)
        monkeypatch.setattr(EINN_PINN, "train_model", spy_train)
        PINNVerificationNode(n_epoch=2, save_plots=False)(verification_state)
        assert seen["init"]["train_size"] == 50  # len(t), not the real-data train_size
        assert seen["init"]["population"] == pytest.approx(1000.0)
        assert seen["init"]["init_params"] == BASE
        assert seen["train"] == {"n_epoch": 2, "lambda_data": 1.0, "lambda_ode": 0.1, "lambda_ic": 0.1, "lambda_bc": 0.0}

    def test_mc_dropout_rate_is_0_005_while_the_log_says_0_05(self, verification_state, monkeypatch, capsys):
        # NOTE: current behavior — possible bug: printed 0.05, passed 0.005
        seen = {}
        original = EINN_PINN.predict_with_uncertainty

        def spy(self, **kwargs):
            seen.update(kwargs)
            return original(self, **kwargs)

        monkeypatch.setattr(EINN_PINN, "predict_with_uncertainty", spy)
        PINNVerificationNode(n_epoch=1, save_plots=False)(verification_state)
        assert seen["dropout_rate"] == 0.005 and seen["n_passes"] == 100
        assert "MC Dropout uncertainty estimation (100 passes, dropout_rate=0.05)" in capsys.readouterr().out

    def test_mean_prediction_replaces_the_deterministic_one(self, verification_state):
        v = PINNVerificationNode(n_epoch=1, save_plots=False)(verification_state)["pinn_verification"]
        assert v["predictions"]["I"] == v["uncertainty"]["mean_I"]
