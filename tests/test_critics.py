"""agents/DeterministicCriticAgent.py (the one in the graph), its two experimental
variants DeterministicCriticAgent2/3, and the LLM-only ParameterCriticAgent.

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
import json
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from agents.DeterministicCriticAgent import DeterministicCriticAgent
from agents.ParameterCriticAgent import ParameterCriticAgent
from tests.support import ScriptedClient, critic_json, intent_json, make_episode

Critic2 = importlib.import_module("agents.DeterministicCriticAgent2").DeterministicCriticAgent
Critic3 = importlib.import_module("agents.DeterministicCriticAgent3").DeterministicCriticAgent

NEW_PARAMS = {"beta": 0.095, "gamma": 0.0553, "mu": 0.0085}


def _results(peak=125.0, height=11000.0, deaths=550.0):
    return {"success": True, "peak_position": peak, "peak_height": height, "total_deaths": deaths}


# =========================================================================== DeterministicCriticAgent


@pytest.fixture
def critic(chat_llm):
    return DeterministicCriticAgent(chat_llm(intent_json()), enable_logging=False)


@pytest.fixture
def pipeline_critic(chat_llm):
    """Exactly how OptimizationPipeline configures the critic."""
    return DeterministicCriticAgent(
        chat_llm(intent_json()),
        enable_logging=True,
        log_format="json",
        max_retries=3,
        position_threshold=1.0,
        height_threshold_relative=0.03,
        position_tolerance=250.0,
        height_tolerance_relative=0.03,
    )


class TestCriticDefaults:
    """Constructor defaults of the critic used by the pipeline, and its unused PromptLogger."""

    def test_defaults(self, critic):
        assert critic.position_threshold == 0.0 and critic.height_threshold_relative == 0.0
        assert critic.position_tolerance == 5.0 and critic.height_tolerance_relative == 0.05
        assert critic.max_retries == 3 and critic.history == [] and critic.task_config == {}
        assert critic.logger is None

    def test_json_logging_creates_a_logger_that_is_never_used(self, pipeline_critic, tmp_path):
        # NOTE: current behavior — PromptLogger is instantiated but no critic method logs anything
        assert pipeline_critic.logger is not None
        pipeline_critic.critique(make_episode(), NEW_PARAMS, _results(), "Need higher peak", "unchanged", "higher")
        assert list((tmp_path / "logs" / "prompts" / "critic").iterdir()) == []

    def test_set_task_config_and_history(self, critic, capsys):
        critic.set_task_config({"a": 1})
        critic.add_to_history(make_episode())
        assert critic.task_config == {"a": 1} and len(critic.history) == 1
        assert "✅ Critic configured" in capsys.readouterr().out

    def test_intent_prompt_variable(self, critic):
        assert critic.intent_prompt.input_variables == ["expert_comment"]


class TestCheckPosition:
    """_check_position(): later / earlier / unchanged branches with thresholds and tolerance."""

    def test_later_ok(self, critic):
        check = critic._check_position(120.0, 125.0, "later")
        assert check == {"ok": True, "description": "moved later: 120.0 → 125.0", "change": 5.0, "is_significant": True}

    def test_later_wrong_direction(self, critic):
        check = critic._check_position(120.0, 115.0, "later")
        assert check["ok"] is False and check["description"] == "expected later, but moved 115.0"

    def test_later_zero_change_is_not_significant(self, critic):
        check = critic._check_position(120.0, 120.0, "later")
        assert check["ok"] is False and check["is_significant"] is False

    def test_earlier(self, critic):
        assert critic._check_position(120.0, 110.0, "earlier")["description"] == "moved earlier: 120.0 → 110.0"
        assert critic._check_position(120.0, 121.0, "earlier")["description"] == "expected earlier, but moved 121.0"

    def test_unchanged_within_default_tolerance(self, critic):
        check = critic._check_position(120.0, 124.0, "unchanged")
        assert check["ok"] is True
        assert check["description"] == "remained stable: 120.0 → 124.0 (Δ=+4.0 within ±5.0)"

    def test_unchanged_outside_tolerance(self, critic):
        check = critic._check_position(120.0, 112.0, "unchanged")
        assert check["ok"] is False
        assert check["description"] == "changed unexpectedly: 120.0 → 112.0 (Δ=-8.0 > ±5.0)"

    def test_unknown_direction_behaves_like_unchanged(self, critic):
        assert critic._check_position(120.0, 121.0, "any")["description"].startswith("remained stable")

    def test_threshold_gates_small_moves(self, chat_llm):
        strict = DeterministicCriticAgent(chat_llm("x"), enable_logging=False, position_threshold=1.0)
        assert strict._check_position(120.0, 120.5, "later")["ok"] is False
        assert strict._check_position(120.0, 121.0, "later")["ok"] is True


class TestCheckHeight:
    """_check_height(): relative-change branches with thresholds and tolerance."""

    def test_higher_ok(self, critic):
        check = critic._check_height(10000.0, 10400.0, "higher")
        assert check == {"ok": True, "description": "increased: 10000 → 10400", "change": 400.0, "is_significant": True}

    def test_higher_wrong(self, critic):
        assert critic._check_height(10000.0, 9000.0, "higher")["description"] == "expected higher, but got 9000"

    def test_lower(self, critic):
        assert critic._check_height(10000.0, 9000.0, "lower")["description"] == "decreased: 10000 → 9000"
        assert critic._check_height(10000.0, 10001.0, "lower")["ok"] is False

    def test_unchanged_relative_tolerance(self, critic):
        ok = critic._check_height(10000.0, 10200.0, "unchanged")
        assert ok["ok"] is True and ok["description"] == "remained stable: 10000 → 10200 (2.0% within ±5%)"
        bad = critic._check_height(10000.0, 11000.0, "unchanged")
        assert bad["ok"] is False and bad["description"] == "changed unexpectedly: 10000 → 11000 (10.0% > ±5%)"

    def test_zero_baseline_has_zero_relative_change(self, critic):
        check = critic._check_height(0.0, 500.0, "unchanged")
        assert check["ok"] is True and check["is_significant"] is True

    def test_relative_threshold(self, chat_llm):
        strict = DeterministicCriticAgent(chat_llm("x"), enable_logging=False, height_threshold_relative=0.03)
        assert strict._check_height(10000.0, 10200.0, "higher")["ok"] is False
        assert strict._check_height(10000.0, 10400.0, "higher")["ok"] is True


class TestMakeDecision:
    """_make_decision(): accept only when both checks pass; never 'adjust'."""

    def test_accept_requires_both(self, critic):
        pos = {"ok": True, "description": "P"}
        height = {"ok": True, "description": "H"}
        assert critic._make_decision(pos, height) == {"decision": "accept", "reasoning": "Position: P. Height: H"}

    def test_reject_lists_only_the_failures(self, critic):
        pos = {"ok": False, "description": "P"}
        height = {"ok": True, "description": "H"}
        assert critic._make_decision(pos, height) == {"decision": "reject", "reasoning": "Position: P"}
        both = critic._make_decision(pos, {"ok": False, "description": "H"})
        assert both == {"decision": "reject", "reasoning": "Position: P; Height: H"}

    def test_never_adjust(self, critic):
        for p in (True, False):
            for h in (True, False):
                d = critic._make_decision({"ok": p, "description": ""}, {"ok": h, "description": ""})["decision"]
                assert d in ("accept", "reject")


class TestParseIntent:
    """_parse_intent(): LLM JSON extraction and the keyword fallback."""

    def test_empty_comment(self, critic):
        assert critic._parse_intent("") == {"position_expected": "unchanged", "height_expected": "unchanged"}

    def test_llm_json(self, chat_llm):
        c = DeterministicCriticAgent(chat_llm(intent_json(True, "earlier", True, "higher")), enable_logging=False)
        assert c._parse_intent("reopen") == {"position_expected": "earlier", "height_expected": "higher"}

    def test_directions_are_used_even_when_cares_flags_are_false(self, chat_llm):
        # NOTE: current behavior — the critic ignores cares_about_* and takes the directions verbatim
        c = DeterministicCriticAgent(chat_llm(intent_json(False, "later", False, "lower")), enable_logging=False)
        assert c._parse_intent("x") == {"position_expected": "later", "height_expected": "lower"}

    def test_json_embedded_in_prose_is_extracted(self, chat_llm):
        c = DeterministicCriticAgent(chat_llm("Sure! " + intent_json(True, "later", True, "lower") + " Done."), enable_logging=False)
        assert c._parse_intent("lockdown") == {"position_expected": "later", "height_expected": "lower"}

    def test_missing_keys_default_to_unchanged(self, chat_llm):
        c = DeterministicCriticAgent(chat_llm('{"foo": 1}'), enable_logging=False)
        assert c._parse_intent("x") == {"position_expected": "unchanged", "height_expected": "unchanged"}

    def test_garbage_falls_back_to_keywords(self, chat_llm, capsys):
        c = DeterministicCriticAgent(chat_llm("no json here"), enable_logging=False)
        assert c._parse_intent("Mask mandate will be introduced") == {"position_expected": "later", "height_expected": "lower"}
        assert "⚠️ LLM failed, using fallback" in capsys.readouterr().out

    def test_llm_exception_falls_back(self):
        boom = RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("down")))
        c = DeterministicCriticAgent(boom, enable_logging=False)
        assert c._parse_intent("Need earlier peak") == {"position_expected": "earlier", "height_expected": "unchanged"}


@pytest.mark.parametrize(
    "comment,expected",
    [
        ("Need later peak", ("later", "unchanged")),
        ("Please delay the wave", ("later", "unchanged")),
        ("Need earlier peak", ("earlier", "unchanged")),
        ("accelerate", ("earlier", "unchanged")),
        ("Need higher peak", ("unchanged", "higher")),
        ("expect a surge", ("unchanged", "higher")),
        ("flatten the curve", ("unchanged", "lower")),
        ("fewer cases", ("unchanged", "lower")),
        ("The peak should be higher and later", ("later", "higher")),
        ("Mask mandate will be introduced", ("later", "lower")),
        ("lockdown but expect a higher peak", ("later", "lower")),  # mitigation words override
        ("Quarantine measures were introduced late", ("later", "lower")),
        ("nothing relevant", ("unchanged", "unchanged")),
        ("LATER", ("later", "unchanged")),
    ],
)
def test_fallback_keyword_table(critic, comment, expected):
    assert critic._fallback_parse(comment) == {"position_expected": expected[0], "height_expected": expected[1]}


class TestCritique:
    """critique(): building the episode, using pre-parsed expectations, history bookkeeping."""

    def test_accept_with_explicit_expectations_skips_the_llm(self, chat_llm):
        llm = chat_llm("would be garbage")
        c = DeterministicCriticAgent(llm, enable_logging=False)
        ep = c.critique(make_episode(), NEW_PARAMS, _results(125.0, 11000.0, 550.0), "Need higher peak", "later", "higher")
        assert ep.accepted is True
        assert ep.reasoning == "Position: moved later: 120.0 → 125.0. Height: increased: 10000 → 11000"
        assert (ep.beta, ep.gamma, ep.mu) == (0.095, 0.0553, 0.0085)
        assert (ep.peak_position, ep.peak_height, ep.total_deaths) == (125.0, 11000.0, 550.0)
        assert ep.iteration == 1 and ep.expert_comment == "Need higher peak"
        assert c.history == [ep]

    def test_iteration_counts_the_critic_history(self, critic):
        critic.history = [make_episode(), make_episode(iteration=1)]
        ep = critic.critique(make_episode(), NEW_PARAMS, _results(), "x", "later", "higher")
        assert ep.iteration == 3 and len(critic.history) == 3

    def test_reject_reasoning(self, critic):
        ep = critic.critique(make_episode(), NEW_PARAMS, _results(110.0, 9000.0), "x", "later", "higher")
        assert ep.accepted is False
        assert ep.reasoning == "Position: expected later, but moved 110.0; Height: expected higher, but got 9000"

    def test_expectations_are_parsed_when_not_supplied(self, chat_llm):
        c = DeterministicCriticAgent(chat_llm(intent_json(True, "later", True, "higher")), enable_logging=False)
        ep = c.critique(make_episode(), NEW_PARAMS, _results(125.0, 11000.0), "x")
        assert ep.accepted is True

    def test_only_one_expectation_supplied_triggers_parsing(self, chat_llm):
        c = DeterministicCriticAgent(chat_llm(intent_json(True, "earlier", True, "lower")), enable_logging=False)
        ep = c.critique(make_episode(), NEW_PARAMS, _results(125.0, 11000.0), "x", position_expected="later")
        assert ep.accepted is False  # parsed (earlier, lower) was used, not the lone "later"

    def test_missing_baseline_metrics_count_as_zero(self, critic):
        baseline = make_episode(peak_position=None, peak_height=None)
        ep = critic.critique(baseline, NEW_PARAMS, _results(10.0, 5.0), "x", "later", "higher")
        assert ep.accepted is True

    def test_console_trace(self, critic, capsys):
        critic.critique(make_episode(), NEW_PARAMS, _results(125.0, 11000.0), "Need higher peak", "unchanged", "higher")
        out = capsys.readouterr().out
        assert "🔍 CRITIC AGENT" in out
        assert "📊 Baseline: peak=120.0, height=10000" in out
        assert "📍 Using pre-parsed expectations (from IntentParser):" in out
        assert "Position: ✓ remained stable" in out
        assert "✅ ACCEPT" in out


class TestPipelineConfiguration:
    """The thresholds OptimizationPipeline hard-codes — including the 250-day tolerance."""

    def test_150_day_shift_is_tolerated_under_unchanged(self, pipeline_critic):
        # NOTE: current behavior — position_tolerance=250 over a 400-day horizon makes the
        # "unchanged" branch accept almost any shift; only the height is really checked
        ep = pipeline_critic.critique(make_episode(), NEW_PARAMS, _results(270.0, 8000.0), "Need lower peak", "unchanged", "lower")
        assert ep.accepted is True
        assert "remained stable: 120.0 → 270.0 (Δ=+150.0 within ±250.0)" in ep.reasoning

    def test_strict_5_day_tolerance_rejects_the_same_shift(self, pipeline_critic):
        pipeline_critic.position_tolerance = 5.0
        ep = pipeline_critic.critique(make_episode(), NEW_PARAMS, _results(270.0, 8000.0), "Need lower peak", "unchanged", "lower")
        assert ep.accepted is False
        assert "changed unexpectedly: 120.0 → 270.0 (Δ=+150.0 > ±5.0)" in ep.reasoning

    def test_position_is_enforced_when_the_expert_asks_for_it(self, pipeline_critic):
        ep = pipeline_critic.critique(make_episode(), NEW_PARAMS, _results(110.0, 8000.0), "later and lower", "later", "lower")
        assert ep.accepted is False and "expected later, but moved 110.0" in ep.reasoning

    def test_three_percent_height_threshold(self, pipeline_critic):
        small = pipeline_critic.critique(make_episode(), NEW_PARAMS, _results(120.0, 10200.0), "higher", "unchanged", "higher")
        assert small.accepted is False
        big = pipeline_critic.critique(make_episode(), NEW_PARAMS, _results(120.0, 10400.0), "higher", "unchanged", "higher")
        assert big.accepted is True

    def test_one_day_position_threshold(self, pipeline_critic):
        assert pipeline_critic.critique(make_episode(), NEW_PARAMS, _results(120.5, 10000.0), "later", "later", "unchanged").accepted is False
        assert pipeline_critic.critique(make_episode(), NEW_PARAMS, _results(121.0, 10000.0), "later", "later", "unchanged").accepted is True


class TestCriticCall:
    """The critic as a graph node: baseline lookup and default expectations."""

    def _state(self, **overrides):
        state = {
            "history": [make_episode()],
            "generated_params": NEW_PARAMS,
            "surrogate_results": _results(125.0, 11000.0),
            "expert_comment": "Need higher peak",
            "expected_position": "unchanged",
            "expected_height": "higher",
        }
        state.update(overrides)
        return state

    def test_call_uses_the_iteration_zero_episode_as_baseline(self, critic):
        state = self._state(history=[make_episode(iteration=5, peak_position=50.0), make_episode(iteration=0)])
        out = critic(state)
        assert out is state
        assert state["critic_decision"] == "accept"
        assert state["critic_reasoning"].startswith("Position: remained stable: 120.0 → 125.0")
        assert state["final_episode"].accepted is True

    def test_call_falls_back_to_current_episode(self, critic):
        state = self._state(history=[], current_episode=make_episode(peak_position=200.0))
        critic(state)
        assert state["critic_decision"] == "reject"  # 200 → 125 is not "unchanged"

    def test_call_without_any_baseline_raises(self, critic):
        with pytest.raises(ValueError, match="No baseline episode"):
            critic(self._state(history=[]))

    def test_call_defaults_expectations_to_unchanged(self, critic):
        state = self._state()
        del state["expected_position"], state["expected_height"]
        critic(state)
        assert state["critic_decision"] == "reject"  # height rose 10 %
        assert "changed unexpectedly: 10000 → 11000" in state["critic_reasoning"]


# =========================================================================== variant 2


class TestCriticVariant2:
    """Experimental variant 2 (agents/DeterministicCriticAgent2.py): no thresholds, accept/reject/adjust."""

    @pytest.fixture
    def c2(self, chat_llm):
        return Critic2(chat_llm("garbage"), enable_logging=False)

    def test_checks_have_no_thresholds(self, c2):
        assert c2._check_position(120.0, 120.0001, "later") == {"ok": True, "description": "moved later: 120.0 → 120.0", "change": pytest.approx(0.0001)}
        assert c2._check_position(120.0, 90.0, "any") == {"ok": True, "description": "changed: 120.0 → 90.0", "change": -30.0}
        assert c2._check_height(10.0, 9.0, "lower")["ok"] is True
        assert c2._check_height(10.0, 9.0, "whatever")["ok"] is True

    def test_decision_matrix(self, c2):
        ok = {"ok": True, "description": "P"}
        bad = {"ok": False, "description": "H"}
        assert c2._make_decision(ok, ok, True, True)["decision"] == "accept"
        assert c2._make_decision(bad, bad, True, True)["decision"] == "reject"
        assert c2._make_decision(ok, bad, True, True)["decision"] == "adjust"
        assert c2._make_decision(bad, bad, False, False)["decision"] == "accept"  # nothing is cared about
        assert c2._make_decision(ok, bad, True, True)["reasoning"] == "Partial success - some metrics need adjustment. Position: P; Height: H"
        assert c2._make_decision(ok, ok, False, False)["reasoning"] == "Parameters changed in expected direction"

    def test_parse_intent_empty_and_llm(self, chat_llm):
        c = Critic2(chat_llm(intent_json(True, "later", False, "any")), enable_logging=False)
        assert c._parse_intent("")["cares_about_position"] is True
        assert c._parse_intent("x")["position_direction"] == "later"

    def test_fallback_treats_the_word_peak_as_position_interest(self, c2):
        # NOTE: current behavior — "peak" is a position keyword, so "Need higher peak"
        # makes the expert care about position with direction "any"
        assert c2._parse_intent("Need higher peak") == {
            "cares_about_position": True,
            "position_direction": "any",
            "cares_about_height": True,
            "height_direction": "higher",
        }
        assert c2._parse_intent("nothing") == {
            "cares_about_position": True,
            "position_direction": "any",
            "cares_about_height": True,
            "height_direction": "any",
        }

    def test_critique_and_call_return_state(self, chat_llm):
        c = Critic2(chat_llm(intent_json(True, "later", True, "higher")), enable_logging=False)
        state = {"history": [make_episode()], "generated_params": NEW_PARAMS, "surrogate_results": _results(125.0, 11000.0), "expert_comment": "x"}
        out = c(state)
        assert out is state and state["critic_decision"] == "accept"
        assert c.history[0].reasoning.startswith("Parameters changed in expected direction")

    def test_adjust_is_not_accepted(self, chat_llm):
        c = Critic2(chat_llm(intent_json(True, "later", True, "higher")), enable_logging=False)
        ep = c.critique(make_episode(), NEW_PARAMS, _results(125.0, 9000.0), "x")
        assert ep.accepted is False and ep.reasoning.startswith("Partial success")


# =========================================================================== variant 3


class TestCriticVariant3:
    """Experimental variant 3 (agents/DeterministicCriticAgent3.py): significance thresholds and their descriptions."""

    @pytest.fixture
    def c3(self, chat_llm):
        return Critic3(chat_llm("garbage"), enable_logging=False, position_threshold=2.0, height_threshold_relative=0.05)

    def test_zero_thresholds_make_every_change_significant(self, chat_llm):
        c = Critic3(chat_llm("x"), enable_logging=False)
        assert c._check_position(120.0, 120.0, "any")["is_significant"] is True
        assert c._check_position(120.0, 120.0, "later")["ok"] is False

    def test_position_descriptions(self, c3):
        assert c3._check_position(120.0, 121.0, "later")["description"] == "moved later but insignificantly: 120.0 → 121.0 (Δ=+1.0 < 2.0 days)"
        assert c3._check_position(120.0, 125.0, "later")["description"] == "moved later significantly: 120.0 → 125.0 (Δ=+5.0 days)"
        assert c3._check_position(120.0, 110.0, "later")["description"] == "moved earlier instead of later: 120.0 → 110.0"
        assert c3._check_position(120.0, 119.0, "earlier")["description"] == "moved earlier but insignificantly: 120.0 → 119.0 (Δ=-1.0 < 2.0 days)"
        assert c3._check_position(120.0, 110.0, "earlier")["ok"] is True
        assert c3._check_position(120.0, 130.0, "earlier")["description"] == "moved later instead of earlier: 120.0 → 130.0"
        assert c3._check_position(120.0, 118.0, "any") == {"ok": True, "description": "changed significantly: 120.0 → 118.0 (Δ=-2.0 days)", "change": -2.0, "is_significant": True}
        assert c3._check_position(120.0, 121.0, "any")["ok"] is False

    def test_height_descriptions(self, c3):
        assert c3._check_height(100.0, 102.0, "higher")["ok"] is False
        assert c3._check_height(100.0, 110.0, "higher")["ok"] is True
        assert c3._check_height(100.0, 90.0, "lower")["ok"] is True
        assert c3._check_height(100.0, 110.0, "lower")["ok"] is False
        assert c3._check_height(100.0, 110.0, "any")["ok"] is True
        assert c3._check_height(0.0, 10.0, "higher")["is_significant"] is False

    def test_insignificant_move_in_the_right_direction_is_adjust_not_reject(self, c3):
        # NOTE: current behavior — _check_* already fold significance into `ok`, so an
        # insignificant move yields ok=False and lands in the "adjust" branch
        pos = c3._check_position(120.0, 121.0, "later")
        height = c3._check_height(100.0, 110.0, "higher")
        d = c3._make_decision(pos, height, True, True)
        assert d["decision"] == "adjust"

    def test_not_significant_enough_branch_is_reachable_only_with_hand_made_checks(self, c3):
        # NOTE: current behavior — dead code for the public checks: ok=True always implies
        # is_significant=True there
        pos = {"ok": True, "description": "P", "is_significant": False}
        height = {"ok": True, "description": "H", "is_significant": True}
        d = c3._make_decision(pos, height, True, True)
        assert d["decision"] == "reject"
        assert d["reasoning"] == "Change was in expected direction but not significant enough. Position: P; Height: H"

    def test_accept_and_adjust(self, c3):
        good = c3._check_position(120.0, 125.0, "later")
        bad = c3._check_position(120.0, 110.0, "later")
        h = c3._check_height(100.0, 110.0, "higher")
        assert c3._make_decision(good, h, True, True)["reasoning"].startswith("Parameters changed significantly in expected direction")
        assert c3._make_decision(bad, h, True, True)["decision"] == "adjust"
        assert c3._make_decision(bad, c3._check_height(100.0, 90.0, "higher"), True, True)["decision"] == "reject"

    def test_task_config_overrides_thresholds(self, c3, capsys):
        c3.set_task_config({"position_threshold": 7.0, "height_threshold_relative": 0.2, "description": "d"})
        assert c3.position_threshold == 7.0 and c3.height_threshold_relative == 0.2
        assert "Thresholds: position=7.0 days, height=20%" in capsys.readouterr().out

    def test_parse_intent_fallback_keys(self, c3):
        assert c3._parse_intent("Need higher peak")["cares_about_height"] is True
        assert c3._parse_intent("")["position_direction"] == "any"

    def test_call_returns_the_mutated_state(self, chat_llm):
        c = Critic3(chat_llm(intent_json(True, "later", True, "higher")), enable_logging=False)
        state = {"history": [make_episode()], "generated_params": NEW_PARAMS, "surrogate_results": _results(125.0, 11000.0), "expert_comment": "x"}
        assert c(state) is state
        assert state["critic_decision"] == "accept" and state["final_episode"].accepted is True

    def test_call_baseline_lookup_and_errors(self, chat_llm, capsys):
        c = Critic3(chat_llm(intent_json(True, "later", True, "higher")), enable_logging=False)
        base = {"generated_params": NEW_PARAMS, "surrogate_results": _results(125.0, 11000.0), "expert_comment": "x"}
        flagged = make_episode(iteration=4)
        flagged.is_baseline = True
        c({**base, "history": [flagged]})
        assert c.history[-1].accepted is True
        c({**base, "history": [], "current_episode": make_episode()})
        assert "⚠️ WARNING: Baseline episode not found" in capsys.readouterr().out
        with pytest.raises(ValueError, match="CRITICAL: No baseline episode"):
            c({**base, "history": []})


# =========================================================================== ParameterCriticAgent


@pytest.fixture
def llm_critic(chat_llm):
    return ParameterCriticAgent(chat_llm(critic_json("accept", "looks right", "high", ["minor"])))


def _critic_logs(tmp_path):
    return sorted((tmp_path / "logs" / "prompts" / "critic").glob("*.json"))


class TestParameterCriticAgent:
    """The LLM-only critic: prompt variables, change formatting, decision paths and its error handlers."""

    def test_prompt_template_variables(self, llm_critic):
        # NOTE: baseline_beta/gamma/mu, new_beta/..., deaths_* are passed in but have no placeholder
        assert set(llm_critic.prompt.input_variables) == {
            "expert_comment",
            "baseline_peak",
            "baseline_height",
            "new_peak",
            "new_height",
            "peak_change",
            "peak_direction",
            "height_change",
            "height_direction",
        }

    def test_format_changes(self, llm_critic):
        cur = {"peak_position": 120.0, "peak_height": 10000.0, "total_deaths": 500.0}
        assert llm_critic._format_changes(cur, {"peak_position": 110.0, "peak_height": 12000.0, "total_deaths": 400.0}) == {
            "peak_change": -10.0,
            "peak_direction": "earlier",
            "height_change": 2000.0,
            "height_direction": "higher",
            "deaths_change": -100.0,
            "deaths_direction": "fewer",
        }
        # zero change is reported as later / higher / more
        assert llm_critic._format_changes(cur, cur) == {
            "peak_change": 0.0,
            "peak_direction": "later",
            "height_change": 0.0,
            "height_direction": "higher",
            "deaths_change": 0.0,
            "deaths_direction": "more",
        }

    def test_accept_path_and_logging(self, llm_critic, tmp_path):
        llm_critic.set_task_config({"description": "d", "target_peak": 100})
        ep = llm_critic.critique(make_episode(), NEW_PARAMS, _results(125.0, 11000.0, 550.0), "Need higher peak")
        assert ep.accepted is True and ep.reasoning == "looks right"
        assert ep.iteration == 1 and llm_critic.history == [ep]
        assert (ep.peak_position, ep.peak_height, ep.total_deaths) == (125.0, 11000.0, 550.0)
        logs = _critic_logs(tmp_path)
        assert len(logs) == 1 and logs[0].name.startswith("critic_iter_001_")
        data = json.loads(logs[0].read_text(encoding="utf-8"))
        assert data["parsed_output"]["decision"] == "accept"
        assert data["metadata"]["decision"] == "accept" and data["metadata"]["confidence"] == "high"
        assert data["context"]["changes"]["peak_direction"] == "later"
        assert data["context"]["new_params"] == NEW_PARAMS
        assert "Expert Comment\n    Need higher peak" in data["prompt"]
        assert "Peak position: +5.0 days (later)" in data["prompt"]

    @pytest.mark.parametrize("decision", ["Reject", "adjust"])
    def test_non_accept_decisions(self, chat_llm, decision):
        c = ParameterCriticAgent(chat_llm(critic_json(decision)), enable_logging=False)
        ep = c.critique(make_episode(), NEW_PARAMS, _results(), "x")
        assert ep.accepted is False and len(c.history) == 1

    def test_adjust_prints_a_notice(self, chat_llm, capsys):
        ParameterCriticAgent(chat_llm(critic_json("adjust")), enable_logging=False).critique(make_episode(), NEW_PARAMS, _results(), "x")
        assert "🔄 Parameters need adjustment according to critic" in capsys.readouterr().out

    def test_no_comment_placeholder(self, llm_critic, tmp_path):
        llm_critic.critique(make_episode(), NEW_PARAMS, _results(), None)
        data = json.loads(_critic_logs(tmp_path)[0].read_text(encoding="utf-8"))
        assert "No expert comment provided." in data["prompt"]

    def test_parse_failure_returns_a_rejected_episode_outside_the_history(self, chat_llm, tmp_path, capsys):
        # NOTE: current behavior — the error-logging attempt builds CriticOutput(confidence=0.0),
        # which fails validation itself, so nothing is logged and a warning is printed
        c = ParameterCriticAgent(chat_llm("not json"))
        ep = c.critique(make_episode(), NEW_PARAMS, _results(), "x")
        assert ep.accepted is False
        assert ep.reasoning.startswith("Critic failed to parse response:")
        assert ep.iteration == 1 and c.history == []
        out = capsys.readouterr().out
        assert "❌ All retry attempts failed" in out
        assert "⚠️ Could not log error" in out
        assert _critic_logs(tmp_path) == []

    def test_llm_exception_returns_a_rejected_episode(self, tmp_path, capsys):
        boom = RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("down")))
        c = ParameterCriticAgent(boom)
        ep = c.critique(make_episode(), NEW_PARAMS, _results(), "x")
        assert ep.accepted is False and ep.reasoning == "Critic error: down"
        out = capsys.readouterr().out
        assert "❌ Critique error: down" in out and "⚠️ Could not log error" in out
        assert _critic_logs(tmp_path) == []

    def test_logging_disabled(self, chat_llm, tmp_path):
        c = ParameterCriticAgent(chat_llm(critic_json()), enable_logging=False)
        assert c.logger is None
        c.critique(make_episode(), NEW_PARAMS, _results(), "x")
        assert not (tmp_path / "logs").exists()

    def test_retry_helpers_with_a_base_client(self):
        client = ScriptedClient(["from client"])
        c = ParameterCriticAgent(client, enable_logging=False)
        assert c.llm_client is client
        assert type(c._get_llm_for_retry()).__name__ == "LLMClientWrapper"
        assert c._call_llm("p") == "from client"

    def test_call_llm_paths(self, chat_llm):
        c = ParameterCriticAgent(chat_llm("from chat"), enable_logging=False)
        assert c._get_llm_for_retry() is c.llm
        assert c._call_llm("p") == "from chat"
        c.llm = object()
        with pytest.raises(ValueError, match="No valid LLM client"):
            c._call_llm("p")

    def test_call_returns_the_mutated_state(self, llm_critic):
        state = {"history": [make_episode()], "generated_params": NEW_PARAMS, "surrogate_results": _results(), "expert_comment": "x"}
        assert llm_critic(state) is state
        assert state["critic_decision"] == "accept" and state["critic_reasoning"] == "looks right"
        assert state["final_episode"].accepted is True

    def test_call_without_baseline_uses_current_episode(self, llm_critic):
        state = {"history": [], "current_episode": make_episode(), "generated_params": NEW_PARAMS, "surrogate_results": _results(), "expert_comment": "x"}
        llm_critic(state)
        assert state["critic_decision"] == "accept"


# =========================================================================== remaining branches


class TestCriticVariant2Extras:
    """Branches of variant 2 not exercised above: json logger, 'earlier' check,
    set_task_config, and the current_episode fallback in __call__."""

    def test_json_logging_creates_a_logger(self, chat_llm):
        assert Critic2(chat_llm("x")).logger is not None
        assert Critic2(chat_llm("x"), log_format="text").logger is None

    def test_earlier_check(self, chat_llm):
        c = Critic2(chat_llm("x"), enable_logging=False)
        assert c._check_position(120.0, 110.0, "earlier") == {"ok": True, "description": "moved earlier: 120.0 → 110.0", "change": -10.0}
        assert c._check_position(120.0, 130.0, "earlier")["ok"] is False

    def test_set_task_config_prints(self, chat_llm, capsys):
        c = Critic2(chat_llm("x"), enable_logging=False)
        c.set_task_config({"description": "demo"})
        assert c.task_config == {"description": "demo"}
        assert "✅ Critic task configured: demo" in capsys.readouterr().out

    def test_call_falls_back_to_current_episode(self, chat_llm):
        c = Critic2(chat_llm(intent_json(True, "later", True, "higher")), enable_logging=False)
        state = {"history": [], "current_episode": make_episode(), "generated_params": NEW_PARAMS, "surrogate_results": _results(125.0, 11000.0), "expert_comment": "x"}
        assert c(state) is state and state["critic_decision"] == "accept"


class TestCriticVariant3Extras:
    """Branches of variant 3 not exercised above, plus the four description branches
    that can never run (ok=False with is_significant=True and a change in the right
    direction is impossible, because ok already requires both)."""

    def test_json_logging_creates_a_logger(self, chat_llm):
        assert Critic3(chat_llm("x")).logger is not None

    def test_insignificant_decrease_description(self, chat_llm):
        c = Critic3(chat_llm("x"), enable_logging=False, height_threshold_relative=0.05)
        check = c._check_height(100.0, 98.0, "lower")
        assert check["ok"] is False
        assert check["description"] == "decreased but insignificantly: 100 → 98 (Δ=-2, 2.0% < 5%)"

    def test_any_direction_insignificant_height(self, chat_llm):
        c = Critic3(chat_llm("x"), enable_logging=False, height_threshold_relative=0.05)
        check = c._check_height(100.0, 101.0, "any")
        assert check["ok"] is False
        assert check["description"] == "no significant change: 100 → 101 (Δ=+1, 1.0% < 5%)"

    def test_fallback_lower_keyword(self, chat_llm):
        c = Critic3(chat_llm("garbage"), enable_logging=False)
        assert c._parse_intent("Need lower peak")["height_direction"] == "lower"
        assert c._parse_intent("decrease it")["height_direction"] == "lower"

    @pytest.mark.parametrize("direction", ["later", "earlier"])
    def test_position_descriptions_never_say_no_significant_change(self, chat_llm, direction):
        # NOTE: current behavior — the "no significant change" else-branches for later/earlier
        # are unreachable: ok=False with a move in the requested direction always means
        # is_significant=False, which is handled by the first branch
        c = Critic3(chat_llm("x"), enable_logging=False, position_threshold=2.0)
        for new in (110.0, 119.0, 120.0, 121.0, 130.0):
            desc = c._check_position(120.0, new, direction)["description"]
            assert not desc.startswith("no significant change")

    @pytest.mark.parametrize("direction", ["higher", "lower"])
    def test_height_descriptions_never_say_no_significant_change(self, chat_llm, direction):
        c = Critic3(chat_llm("x"), enable_logging=False, height_threshold_relative=0.05)
        for new in (80.0, 98.0, 100.0, 102.0, 120.0):
            desc = c._check_height(100.0, new, direction)["description"]
            assert not desc.startswith("no significant change")


class TestParameterCriticWrapper:
    """The LangChain wrapper ParameterCriticAgent builds around a BaseLLMClient."""

    def test_wrapper_methods(self):
        client = ScriptedClient(["from client"])
        wrapper = ParameterCriticAgent(client, enable_logging=False)._get_llm_for_retry()
        assert wrapper._llm_type == "base_llm_client_wrapper"
        assert wrapper._identifying_params == {"model": "scripted-model"}
        assert wrapper.invoke("hello") == "from client"
        assert client.prompts == ["hello"]
