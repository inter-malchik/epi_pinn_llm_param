"""formats/data_formats.py — Pydantic contracts, Episode, and the PipelineState schema.

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

import pytest
from pydantic import ValidationError

from formats.data_formats import CriticOutput, EpiParameters, Episode, ExpertIntent, PipelineState
from tests.conftest import PROJECT_ROOT
from tests.support import make_episode

# --------------------------------------------------------------------------- EpiParameters


def _params(**overrides):
    fields = dict(beta=0.1, gamma=0.05, mu=0.005, reasoning="ok", confidence="high")
    fields.update(overrides)
    return EpiParameters(**fields)


class TestEpiParameters:
    """Validation rules of the generator's output model, including the β range that is wider than its own error message claims."""

    def test_valid_instance_keeps_values(self):
        p = _params()
        assert (p.beta, p.gamma, p.mu, p.reasoning, p.confidence) == (0.1, 0.05, 0.005, "ok", "high")

    @pytest.mark.parametrize("beta", [0.0, 1.0, 1.5, 2.5, 3.0])
    def test_beta_accepts_values_up_to_three(self, beta):
        # NOTE: current behavior — the docstring says 0.0–1.0 but the validator allows up to 3.0
        assert _params(beta=beta).beta == beta

    @pytest.mark.parametrize("beta", [-0.01, 3.01, 3.5])
    def test_beta_out_of_range_error_message_claims_a_narrower_range(self, beta):
        # NOTE: current behavior — possible bug: the message says "between 0.0 and 1.0"
        # although the check is 0.0 <= beta <= 3.0
        with pytest.raises(ValidationError) as excinfo:
            _params(beta=beta)
        assert f"beta must be between 0.0 and 1.0, got {beta}" in str(excinfo.value)

    @pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
    def test_gamma_bounds_inclusive(self, gamma):
        assert _params(gamma=gamma).gamma == gamma

    @pytest.mark.parametrize("gamma", [-0.1, 1.01])
    def test_gamma_out_of_range(self, gamma):
        with pytest.raises(ValidationError) as excinfo:
            _params(gamma=gamma)
        assert "gamma must be between 0.00 and 1.0" in str(excinfo.value)

    @pytest.mark.parametrize("mu", [0.0, 0.05, 0.1])
    def test_mu_bounds_inclusive(self, mu):
        assert _params(mu=mu).mu == mu

    @pytest.mark.parametrize("mu", [-0.001, 0.1001, 0.5])
    def test_mu_out_of_range(self, mu):
        with pytest.raises(ValidationError) as excinfo:
            _params(mu=mu)
        assert "mu must be between 0.000 and 0.1" in str(excinfo.value)

    @pytest.mark.parametrize("confidence", ["high", "medium", "low"])
    def test_confidence_levels(self, confidence):
        assert _params(confidence=confidence).confidence == confidence

    @pytest.mark.parametrize("confidence", ["High", "HIGH", "certain", ""])
    def test_confidence_is_case_sensitive(self, confidence):
        with pytest.raises(ValidationError) as excinfo:
            _params(confidence=confidence)
        assert "confidence must be high/medium/low" in str(excinfo.value)

    def test_numeric_strings_are_coerced(self):
        p = _params(beta="0.25", gamma="0.1", mu="0.01")
        assert (p.beta, p.gamma, p.mu) == (0.25, 0.1, 0.01)

    def test_non_numeric_beta_rejected(self):
        with pytest.raises(ValidationError):
            _params(beta="high")

    @pytest.mark.parametrize("missing", ["beta", "gamma", "mu", "reasoning", "confidence"])
    def test_every_field_is_required(self, missing):
        fields = dict(beta=0.1, gamma=0.05, mu=0.005, reasoning="ok", confidence="high")
        del fields[missing]
        with pytest.raises(ValidationError):
            EpiParameters(**fields)

    def test_extra_fields_are_ignored(self):
        p = EpiParameters(beta=0.1, gamma=0.05, mu=0.005, reasoning="ok", confidence="low", expected_peak_position=30)
        assert not hasattr(p, "expected_peak_position")


# --------------------------------------------------------------------------- Episode


class TestEpisode:
    """The dataclass that records one optimization step and renders itself into the generator prompt."""

    def test_required_and_default_fields(self):
        ep = Episode(beta=0.1, gamma=0.05, mu=0.005, reasoning=None)
        assert ep.peak_position is None
        assert ep.peak_height is None
        assert ep.total_deaths is None
        assert ep.expert_comment is None
        assert ep.accepted is False
        assert ep.iteration is None

    def test_timestamp_is_filled_in_post_init(self):
        ep = Episode(beta=0.1, gamma=0.05, mu=0.005, reasoning=None)
        assert isinstance(ep.timestamp, str)
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ep.timestamp)

    def test_explicit_timestamp_is_kept(self):
        ep = Episode(beta=0.1, gamma=0.05, mu=0.005, reasoning=None, timestamp="2026-01-01T00:00:00")
        assert ep.timestamp == "2026-01-01T00:00:00"

    def test_to_dict_contains_every_field(self):
        ep = make_episode(timestamp="ts")
        assert ep.to_dict() == {
            "beta": 0.091,
            "gamma": 0.0553,
            "mu": 0.0085,
            "reasoning": "Baseline parameters from initial input",
            "peak_position": 120.0,
            "peak_height": 10000.0,
            "total_deaths": 500.0,
            "timestamp": "ts",
            "expert_comment": "Need higher peak",
            "accepted": True,
            "iteration": 0,
        }

    def test_prompt_format_accepted_with_reasoning(self):
        ep = make_episode(iteration=3, beta=0.0952, mu=0.0085, peak_position=128.44, peak_height=41207.4, total_deaths=3890.6)
        assert ep.to_prompt_format() == (
            "**Iteration 3:**\n"
            "- Parameters: β=0.0952, γ=0.0553, μ=0.00850\n"
            "- Results: peak at day 128.4, height 41207, deaths 3891\n"
            "- Status: ✓ ACCEPTED\n"
            "- Expert comment: Need higher peak\n"
            "- Reasoning: Baseline parameters from initial input\n"
        )

    def test_prompt_format_rejected_without_reasoning_or_comment(self):
        ep = make_episode(iteration=1, accepted=False, reasoning=None, expert_comment=None)
        text = ep.to_prompt_format()
        assert "- Status: ✗ REJECTED\n" in text
        assert "- Expert comment: None\n" in text
        assert "Reasoning" not in text

    def test_prompt_format_empty_reasoning_is_omitted(self):
        assert "Reasoning" not in make_episode(reasoning="").to_prompt_format()

    def test_prompt_format_requires_numeric_metrics(self):
        # NOTE: current behavior — an Episode without surrogate metrics cannot be rendered
        with pytest.raises(TypeError):
            Episode(beta=0.1, gamma=0.05, mu=0.005, reasoning=None, iteration=1).to_prompt_format()

    def test_equality_ignores_nothing(self):
        a = make_episode(timestamp="ts")
        b = make_episode(timestamp="ts")
        assert a == b
        assert a != make_episode(timestamp="other")


# --------------------------------------------------------------------------- CriticOutput


class TestCriticOutput:
    """Output model of the LLM-only critic: decision normalization and defaults."""

    @pytest.mark.parametrize("raw,expected", [("accept", "accept"), ("Reject", "reject"), ("ADJUST", "adjust"), ("  Accept ", "accept")])
    def test_decision_is_normalized(self, raw, expected):
        assert CriticOutput(reasoning="r", decision=raw).decision == expected

    @pytest.mark.parametrize("raw", ["approve", "", "accept!"])
    def test_unknown_decision_rejected(self, raw):
        with pytest.raises(ValidationError) as excinfo:
            CriticOutput(reasoning="r", decision=raw)
        assert "decision must be accept/reject/adjust" in str(excinfo.value)

    def test_non_string_decision_rejected(self):
        with pytest.raises(ValidationError):
            CriticOutput(reasoning="r", decision=1)

    def test_defaults(self):
        out = CriticOutput(reasoning="r", decision="accept")
        assert out.confidence == "medium"
        assert out.issues == []

    def test_confidence_is_not_validated(self):
        assert CriticOutput(reasoning="r", decision="accept", confidence="banana").confidence == "banana"

    def test_float_confidence_is_rejected(self):
        # NOTE: ParameterCriticAgent builds CriticOutput(confidence=0.0) in its error
        # handlers — that construction fails validation and is swallowed there.
        with pytest.raises(ValidationError):
            CriticOutput(reasoning="r", decision="reject", confidence=0.0)

    def test_issues_list_kept(self):
        assert CriticOutput(reasoning="r", decision="adjust", issues=["a", "b"]).issues == ["a", "b"]


# --------------------------------------------------------------------------- ExpertIntent


class TestExpertIntent:
    """The parsed expert intent: only primary_metric is a constrained Literal."""

    def _intent(self, **overrides):
        fields = dict(
            cares_about_position=True,
            position_direction="later",
            cares_about_height=False,
            height_direction="any",
            reasoning="r",
        )
        fields.update(overrides)
        return ExpertIntent(**fields)

    def test_primary_metric_defaults_to_both(self):
        assert self._intent().primary_metric == "both"

    @pytest.mark.parametrize("metric", ["position", "height", "both"])
    def test_primary_metric_literal(self, metric):
        assert self._intent(primary_metric=metric).primary_metric == metric

    def test_primary_metric_outside_literal_rejected(self):
        with pytest.raises(ValidationError):
            self._intent(primary_metric="deaths")

    def test_directions_are_free_strings(self):
        # NOTE: current behavior — only primary_metric is constrained; directions are plain str
        intent = self._intent(position_direction="sideways", height_direction="wider")
        assert (intent.position_direction, intent.height_direction) == ("sideways", "wider")

    @pytest.mark.parametrize("raw,expected", [("true", True), ("yes", True), ("0", False), (1, True)])
    def test_boolean_flags_are_coerced(self, raw, expected):
        assert self._intent(cares_about_position=raw).cares_about_position is expected

    def test_reasoning_required(self):
        with pytest.raises(ValidationError):
            ExpertIntent(cares_about_position=True, position_direction="later", cares_about_height=True, height_direction="lower")


# --------------------------------------------------------------------------- PipelineState

DECLARED_STATE_KEYS = {
    "task_config",
    "current_episode",
    "expert_comment",
    "expected_position",
    "expected_height",
    "history",
    "generated_params",
    "surrogate_results",
    "critic_decision",
    "critic_reasoning",
    "final_episode",
    "pinn_results",
    "iteration",
    "max_iterations",
    "should_continue",
}

# Keys the nodes read or write through `state[...]` / `state.get(...)` but which
# PipelineState does not declare. LangGraph builds its channels from the TypedDict
# annotations only, so values stored under these keys do not survive a node hop
# (see test_pipeline_graph.py for the runtime consequence).
UNDECLARED_STATE_KEYS = {
    "sensitivity_map",
    "initial_conditions",
    "simulation_params",
    "expert_intent",
    "pinn_verification",
    "direction_hint",
    "generation_error",
    "peak_error",
    "is_acceptable",
    "pinn_data",
}

_STATE_ACCESS = re.compile(r"""\bstate(?:\[|\.get\()\s*['"](\w+)['"]""")


def _keys_used_in_sources():
    sources = [PROJECT_ROOT / "main_test.py", *sorted((PROJECT_ROOT / "agents").glob("*.py"))]
    found = set()
    for path in sources:
        found.update(_STATE_ACCESS.findall(path.read_text(encoding="utf-8")))
    return found


class TestPipelineState:
    """Which keys the LangGraph state schema declares versus which keys the nodes actually use (scanned from the source files)."""

    def test_declared_keys_exact(self):
        assert set(PipelineState.__annotations__) == DECLARED_STATE_KEYS

    def test_every_key_used_by_nodes_is_either_declared_or_in_the_known_gap(self):
        # NOTE: current behavior — possible bug: the UNDECLARED set is non-empty.
        # If you add a key to PipelineState, remove it from UNDECLARED_STATE_KEYS here.
        used = _keys_used_in_sources()
        assert used - DECLARED_STATE_KEYS == UNDECLARED_STATE_KEYS
        # `should_continue` is declared and seeded in run(), but no node ever reads or writes it
        assert DECLARED_STATE_KEYS - used == {"should_continue"}

    def test_undeclared_keys_are_really_missing_from_the_schema(self):
        assert not (UNDECLARED_STATE_KEYS & set(PipelineState.__annotations__))
