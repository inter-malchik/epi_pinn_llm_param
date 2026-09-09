"""agents/SurrogateModel.py — classical SIRD on scipy and the SurrogateAgent node.

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
import numpy as np
import pytest

from agents.SurrogateModel import SIRDSurrogate, SurrogateAgent

# --------------------------------------------------------------------------- SIRDSurrogate


class TestSIRDSurrogate:
    """Numerical properties of the scipy SIRD integration: conservation, monotonicity, peak metrics and the two failure modes of simulate()."""

    def test_ode_system_conserves_population(self):
        model = SIRDSurrogate(population=1000, S0=999, I0=1, R0=0, D0=0)
        dS, dI, dR, dD = model.ode_system(0.0, [900.0, 50.0, 40.0, 10.0], 0.1, 0.06, 0.003)
        assert dS == pytest.approx(-0.1 * 900 * 50 / 1000)
        assert dI == pytest.approx(0.1 * 900 * 50 / 1000 - 0.06 * 50 - 0.003 * 50)
        assert dR == pytest.approx(0.06 * 50)
        assert dD == pytest.approx(0.003 * 50)
        assert dS + dI + dR + dD == pytest.approx(0.0)

    def test_simulate_returns_full_trajectories(self):
        model = SIRDSurrogate(population=1000, S0=999, I0=1, R0=0, D0=0)
        res = model.simulate(beta=0.1, gamma=0.06, mu=0.003, t_max=400, num_points=1000)
        assert set(res) == {
            "peak_position", "peak_height", "peak_day", "t", "I", "S", "R", "D",
            "final_infected", "total_recovered", "total_deaths",
        }
        assert len(res["t"]) == 1000
        assert res["t"][0] == 0.0 and res["t"][-1] == 400.0
        total = res["S"] + res["I"] + res["R"] + res["D"]
        assert np.allclose(total, 1000.0, rtol=1e-4)

    def test_simulate_peak_metrics_are_consistent(self):
        model = SIRDSurrogate(population=1000, S0=999, I0=1, R0=0, D0=0)
        res = model.simulate(beta=0.1, gamma=0.06, mu=0.003, t_max=400, num_points=1000)
        idx = int(np.argmax(res["I"]))
        assert res["peak_position"] == res["t"][idx]
        assert res["peak_height"] == res["I"][idx]
        assert res["peak_day"] == int(res["peak_position"])
        # ground-truth scenario of 01_baseline_constant.csv: R0≈1.59 → peak ≈ day 161, ≈ 8% of N
        assert res["peak_position"] == pytest.approx(161.0, abs=1.0)
        assert res["peak_height"] == pytest.approx(78.0, abs=3.0)
        assert res["total_deaths"] == res["D"][-1]
        assert res["total_recovered"] == res["R"][-1]
        assert res["final_infected"] == res["I"][-1]

    def test_simulate_monotone_compartments(self):
        model = SIRDSurrogate(population=1000, S0=999, I0=1, R0=0, D0=0)
        res = model.simulate(beta=0.1, gamma=0.06, mu=0.003, t_max=400, num_points=1000)
        assert np.all(np.diff(res["S"]) <= 1e-9)
        assert np.all(np.diff(res["R"]) >= -1e-9)
        assert np.all(np.diff(res["D"]) >= -1e-9)

    def test_subcritical_r0_peaks_at_start(self):
        model = SIRDSurrogate(population=1000, S0=999, I0=1, R0=0, D0=0)
        res = model.simulate(beta=0.05, gamma=0.06, mu=0.003, t_max=200, num_points=500)
        assert res["peak_position"] == 0.0
        assert res["peak_height"] == pytest.approx(1.0)
        assert np.all(np.diff(res["I"]) <= 0)
        # decay rate γ+μ-β = 0.013/day → I(200) ≈ e^-2.6 ≈ 0.07
        assert res["final_infected"] == pytest.approx(0.0717, abs=0.005)

    def test_higher_beta_gives_earlier_and_higher_peak(self):
        model = SIRDSurrogate(population=1000, S0=999, I0=1, R0=0, D0=0)
        base = model.simulate(0.1, 0.06, 0.003, t_max=400, num_points=1000)
        more = model.simulate(0.12, 0.06, 0.003, t_max=400, num_points=1000)
        assert more["peak_position"] < base["peak_position"]
        assert more["peak_height"] > base["peak_height"]

    def test_mu_barely_moves_the_peak_but_changes_deaths(self):
        model = SIRDSurrogate(population=1000, S0=999, I0=1, R0=0, D0=0)
        base = model.simulate(0.1, 0.06, 0.003, t_max=400, num_points=1000)
        deadly = model.simulate(0.1, 0.06, 0.006, t_max=400, num_points=1000)
        assert abs(deadly["peak_position"] - base["peak_position"]) < 10
        assert deadly["total_deaths"] > 1.5 * base["total_deaths"]

    def test_t_max_none_raises_before_the_try_block(self):
        # NOTE: current behavior — t_eval is built outside the try/except, so a missing
        # t_max is a TypeError, not a `None` result
        model = SIRDSurrogate(population=1000, S0=999, I0=1, R0=0, D0=0)
        with pytest.raises(TypeError):
            model.simulate(beta=0.1, gamma=0.06, mu=0.003)

    def test_solver_failure_returns_none_and_prints(self, capsys):
        model = SIRDSurrogate(population=1000, S0=999, I0=1, R0=0, D0=0)
        assert model.simulate(beta=0.1, gamma=0.06, mu=0.003, t_max=100, num_points=0) is None
        assert "Ошибка симуляции для β=0.100" in capsys.readouterr().out


# --------------------------------------------------------------------------- SurrogateAgent.simulate


class TestSurrogateAgentSimulate:
    """SurrogateAgent.simulate(): plain-Python payload on success, structured error on failure."""

    def test_success_payload_uses_plain_python_types(self, surrogate_agent, initial_conditions):
        res = surrogate_agent.simulate(0.1, 0.06, 0.003, **initial_conditions, t_max=400, num_points=500)
        assert res["success"] is True
        for key in ("beta", "gamma", "mu", "peak_position", "peak_height", "total_recovered", "total_deaths", "final_infected"):
            assert type(res[key]) is float, key
        assert type(res["peak_day"]) is int
        for key in ("t", "I", "S", "R", "D"):
            assert isinstance(res[key], list) and len(res[key]) == 500
        assert res["beta"] == 0.1 and res["gamma"] == 0.06 and res["mu"] == 0.003

    def test_defaults_are_200_days_and_1000_points(self, surrogate_agent, initial_conditions):
        res = surrogate_agent.simulate(0.1, 0.06, 0.003, **initial_conditions)
        assert len(res["t"]) == 1000
        assert res["t"][-1] == 200.0

    def test_failure_payload(self, surrogate_agent, initial_conditions):
        res = surrogate_agent.simulate(0.1, 0.06, 0.003, **initial_conditions, t_max=100, num_points=0)
        assert res == {"success": False, "error": "Симуляция не удалась", "beta": 0.1, "gamma": 0.06, "mu": 0.003}

    def test_cache_attribute_exists_but_is_never_used(self, surrogate_agent, initial_conditions):
        surrogate_agent.simulate(0.1, 0.06, 0.003, **initial_conditions)
        assert surrogate_agent.cache == {}


# --------------------------------------------------------------------------- SurrogateAgent.__call__


def _state(ic, **overrides):
    state = {
        "generated_params": {"beta": 0.1, "gamma": 0.06, "mu": 0.003},
        "initial_conditions": dict(ic),
    }
    state.update(overrides)
    return state


class TestSurrogateAgentCall:
    """SurrogateAgent as a graph node: input validation, defaults and the optional target-peak evaluation."""

    def test_missing_generated_params(self, surrogate_agent, initial_conditions):
        out = surrogate_agent(_state(initial_conditions, generated_params={}))
        assert out["surrogate_results"] == {"success": False, "error": "Нет параметров для симуляции"}

    def test_missing_initial_conditions(self, surrogate_agent, initial_conditions):
        out = surrogate_agent(_state(initial_conditions, initial_conditions={}))
        assert out["surrogate_results"]["success"] is False
        assert out["surrogate_results"]["error"].startswith("Нет начальных условий")

    def test_incomplete_parameters(self, surrogate_agent, initial_conditions):
        out = surrogate_agent(_state(initial_conditions, generated_params={"beta": 0.1, "gamma": 0.06}))
        assert out["surrogate_results"] == {"success": False, "error": "Неполные параметры: beta=0.1, gamma=0.06, mu=None"}

    def test_incomplete_initial_conditions(self, surrogate_agent, initial_conditions):
        ic = dict(initial_conditions)
        del ic["S0"]
        out = surrogate_agent(_state(initial_conditions, initial_conditions=ic))
        assert out["surrogate_results"] == {"success": False, "error": "Неполные начальные условия"}

    def test_zero_beta_is_a_valid_parameter_here(self, surrogate_agent, initial_conditions):
        out = surrogate_agent(_state(initial_conditions, generated_params={"beta": 0.0, "gamma": 0.06, "mu": 0.003}))
        assert out["surrogate_results"]["success"] is True
        assert out["surrogate_results"]["peak_position"] == 0.0

    def test_defaults_when_no_simulation_params(self, surrogate_agent, initial_conditions):
        out = surrogate_agent(_state(initial_conditions))
        res = out["surrogate_results"]
        assert res["success"] is True
        assert len(res["t"]) == 1000
        assert res["t"][-1] == 400.0

    def test_simulation_params_are_honoured(self, surrogate_agent, initial_conditions):
        out = surrogate_agent(_state(initial_conditions, simulation_params={"t_max": 50, "num_points": 20}))
        res = out["surrogate_results"]
        assert len(res["t"]) == 20 and res["t"][-1] == 50.0

    def test_state_is_mutated_and_returned(self, surrogate_agent, initial_conditions):
        state = _state(initial_conditions)
        out = surrogate_agent(state)
        assert out is state
        assert "surrogate_results" in state

    def test_target_peak_evaluation_with_default_tolerance(self, surrogate_agent, initial_conditions, baseline_results):
        peak = baseline_results["peak_position"]
        out = surrogate_agent(_state(initial_conditions, task_config={"target_peak": peak + 3.0}))
        assert out["peak_error"] == pytest.approx(3.0)
        assert out["is_acceptable"] is True
        out = surrogate_agent(_state(initial_conditions, task_config={"target_peak": peak + 6.0}))
        assert out["is_acceptable"] is False

    def test_target_peak_custom_tolerance(self, surrogate_agent, initial_conditions, baseline_results):
        peak = baseline_results["peak_position"]
        out = surrogate_agent(_state(initial_conditions, task_config={"target_peak": peak + 6.0, "peak_tolerance": 10}))
        assert out["is_acceptable"] is True

    def test_no_evaluation_without_target_peak(self, surrogate_agent, initial_conditions):
        out = surrogate_agent(_state(initial_conditions, task_config={}))
        assert "peak_error" not in out and "is_acceptable" not in out

    def test_no_evaluation_when_simulation_failed(self, surrogate_agent, initial_conditions):
        out = surrogate_agent(_state(initial_conditions, simulation_params={"t_max": 10, "num_points": 0}, task_config={"target_peak": 5}))
        assert out["surrogate_results"]["success"] is False
        assert "peak_error" not in out

    def test_verbose_prints_r0(self, initial_conditions, capsys):
        SurrogateAgent(verbose=True)(_state(initial_conditions))
        out = capsys.readouterr().out
        assert "R0 (basic reproduction): 1.59" in out
        assert "SURROGATE AGENT" in out

    def test_quiet_mode_prints_nothing(self, surrogate_agent, initial_conditions, capsys):
        surrogate_agent(_state(initial_conditions))
        assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- verbose branches


class TestSurrogateAgentVerbose:
    """Console messages printed by SurrogateAgent(verbose=True) on every error branch."""

    @pytest.fixture
    def loud(self):
        return SurrogateAgent(verbose=True)

    def test_missing_params_message(self, loud, initial_conditions, capsys):
        loud(_state(initial_conditions, generated_params={}))
        assert "❌ Нет параметров для симуляции" in capsys.readouterr().out

    def test_missing_initial_conditions_message(self, loud, initial_conditions, capsys):
        loud(_state(initial_conditions, initial_conditions={}))
        assert "❌ Нет начальных условий в state" in capsys.readouterr().out

    def test_incomplete_params_message(self, loud, initial_conditions, capsys):
        loud(_state(initial_conditions, generated_params={"beta": 0.1}))
        assert "❌ Неполные параметры: beta=0.1, gamma=None, mu=None" in capsys.readouterr().out

    def test_incomplete_initial_conditions_message(self, loud, initial_conditions, capsys):
        loud(_state(initial_conditions, initial_conditions={"population": 1000}))
        assert "❌ Неполные начальные условия: population=1000, S0=None, I0=None, R0=None, D0=None" in capsys.readouterr().out

    def test_simulation_failure_message(self, loud, initial_conditions, capsys):
        loud(_state(initial_conditions, simulation_params={"t_max": 10, "num_points": 0}))
        assert "❌ Ошибка симуляции: Симуляция не удалась" in capsys.readouterr().out

    def test_target_peak_message(self, loud, initial_conditions, baseline_results, capsys):
        peak = baseline_results["peak_position"]
        loud(_state(initial_conditions, task_config={"target_peak": peak + 1.0}))
        out = capsys.readouterr().out
        assert f"🎯 Оценка относительно целевого пика (день {peak + 1.0}):" in out
        assert "Ошибка: 1.0 дней" in out and "Приемлемо: ✅ Да" in out
