"""agents/IntentParserAgent.py and agents/EpiParamGeneratorAgent.py.

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
import types
from pathlib import Path

import pytest
from langchain_core.exceptions import OutputParserException

from agents.EpiParamGeneratorAgent import LLMEpiParamGenerator
from agents.IntentParserAgent import IntentParserAgent
from agents.SurrogateModel import SurrogateAgent
from formats.data_formats import ExpertIntent
from main_test import SensitivityNode
from tests.support import ScriptedClient, intent_json, make_episode, params_json

# --------------------------------------------------------------------------- IntentParserAgent


class TestIntentParserAgent:
    """IntentParserAgent: prompt shape, empty-comment shortcut, JSON parsing and the absence of any fallback."""

    def test_prompt_has_a_single_input_variable(self, chat_llm):
        agent = IntentParserAgent(chat_llm("x"))
        assert agent.prompt.input_variables == ["expert_comment"]
        messages = agent.prompt.format_messages(expert_comment="Need higher peak")
        assert messages[-1].content == "Expert comment: Need higher peak"
        system = messages[0].content
        assert "Mask mandate will be introduced" in system  # few-shot example
        assert "cares_about_position" in system  # format instructions were injected

    @pytest.mark.parametrize("comment", ["", None])
    def test_empty_comment_short_circuits_without_the_llm(self, chat_llm, comment):
        llm = chat_llm("should never be used")
        intent = IntentParserAgent(llm).parse(comment)
        assert intent == ExpertIntent(
            cares_about_position=False,
            position_direction="any",
            cares_about_height=False,
            height_direction="any",
            primary_metric="both",
            reasoning="No comment provided",
        )

    def test_valid_json_is_parsed(self, chat_llm):
        intent = IntentParserAgent(chat_llm(intent_json(True, "later", True, "lower", "height", "masks"))).parse("Mask mandate")
        assert (intent.position_direction, intent.height_direction, intent.primary_metric, intent.reasoning) == ("later", "lower", "height", "masks")

    def test_fenced_json_is_parsed(self, chat_llm):
        reply = "```json\n" + intent_json(False, "any", True, "higher") + "\n```"
        intent = IntentParserAgent(chat_llm(reply)).parse("Need higher peak")
        assert intent.cares_about_position is False and intent.height_direction == "higher"

    def test_truncated_json_is_repaired_by_the_parser(self, chat_llm):
        truncated = intent_json(True, "earlier", False, "any")[:-1]  # drop the closing brace
        intent = IntentParserAgent(chat_llm(truncated)).parse("sooner")
        assert intent.position_direction == "earlier"

    @pytest.mark.parametrize(
        "reply",
        [
            "Garbage response that is definitely not JSON",
            "Sure! Here is the JSON: " + intent_json(),
            '{"cares_about_position": true}',  # missing required fields
            intent_json(primary="deaths"),  # outside the Literal
        ],
    )
    def test_unparseable_reply_raises_instead_of_falling_back(self, chat_llm, reply):
        # NOTE: current behavior — possible bug: no try/except, no retry, no keyword
        # fallback; the exception stops the whole graph
        with pytest.raises(OutputParserException):
            IntentParserAgent(chat_llm(reply)).parse("Lockdown has been introduced")

    def test_call_maps_directions_into_state(self, chat_llm):
        agent = IntentParserAgent(chat_llm(intent_json(True, "later", True, "lower")))
        state = {"expert_comment": "Lockdown"}
        out = agent(state)
        assert out is state
        assert state["expected_position"] == "later"
        assert state["expected_height"] == "lower"
        assert isinstance(state["expert_intent"], ExpertIntent)

    def test_call_uses_unchanged_when_the_expert_does_not_care(self, chat_llm):
        agent = IntentParserAgent(chat_llm(intent_json(False, "later", True, "higher")))
        state = agent({"expert_comment": "Need higher peak"})
        # NOTE: the direction the model wrote ("later") is discarded when cares_about_position is False
        assert state["expected_position"] == "unchanged"
        assert state["expected_height"] == "higher"

    def test_call_without_comment_is_neutral(self, chat_llm):
        state = IntentParserAgent(chat_llm("unused"))({})
        assert (state["expected_position"], state["expected_height"]) == ("unchanged", "unchanged")

    def test_call_prints_the_parsed_intent(self, chat_llm, capsys):
        IntentParserAgent(chat_llm(intent_json(True, "later", False, "any", "position", "why")))({"expert_comment": "later"})
        out = capsys.readouterr().out
        assert "🎯 INTENT PARSER AGENT (LLM only)" in out
        assert "cares_about_position: True → later" in out
        assert "cares_about_height:   False → any" in out
        assert "primary_metric: position" in out
        assert "reasoning: why" in out


# --------------------------------------------------------------------------- generator: construction & formatting


@pytest.fixture
def generator(chat_llm):
    return LLMEpiParamGenerator(chat_llm(params_json(0.095, 0.0553, 0.0085)))


@pytest.fixture
def sensitivity_map(initial_conditions):
    node = SensitivityNode(SurrogateAgent(verbose=False))
    state = node({"task_config": {"beta": 0.1, "gamma": 0.06, "mu": 0.003, **initial_conditions}, "simulation_params": {"t_max": 400, "num_points": 500}})
    return state["sensitivity_map"]


class TestGeneratorConstruction:
    """LLMEpiParamGenerator construction: LLM wrapping, logging flags, prompt template variables."""

    def test_langchain_llm_is_used_directly(self, chat_llm):
        llm = chat_llm("x")
        gen = LLMEpiParamGenerator(llm)
        assert gen.llm is llm and gen.llm_client is None
        assert gen.history == [] and gen.task_config == {}
        assert gen.max_retries == 3 and gen.retry_temperature == 0.3
        assert gen.retry_parser.max_retries == 3

    def test_base_client_is_kept_and_wrapped_for_retries(self):
        client = ScriptedClient(["x"])
        gen = LLMEpiParamGenerator(client)
        assert gen.llm_client is client and gen.llm is client
        wrapper = gen._get_llm_for_retry()
        assert type(wrapper).__name__ == "LLMClientWrapper"
        assert wrapper._llm_type == "base_llm_client_wrapper"
        assert wrapper._identifying_params == {"model": "scripted-model"}
        assert wrapper.invoke("hello") == "x"
        assert client.prompts == ["hello"]

    def test_raw_base_client_cannot_drive_generate(self):
        # NOTE: current behavior — generate() builds `RunnableParallel(response=self.llm)`;
        # a BaseLLMClient is not a Runnable, so only the pipeline's pre-wrapped LLM works here
        gen = LLMEpiParamGenerator(ScriptedClient([params_json(0.1, 0.05, 0.005)]))
        with pytest.raises(TypeError):
            gen.generate({"expert_comment": "x", "current_episode": make_episode(), "iteration": 0})

    def test_call_llm_paths(self, chat_llm):
        gen = LLMEpiParamGenerator(ScriptedClient(["from client"]))
        assert gen._call_llm("p") == "from client"
        assert LLMEpiParamGenerator(chat_llm("from chat"))._call_llm("p") == "from chat"
        gen.llm_client = None
        gen.llm = object()
        with pytest.raises(ValueError, match="No valid LLM client"):
            gen._call_llm("p")

    def test_logging_flags(self, chat_llm):
        assert LLMEpiParamGenerator(chat_llm("x"), enable_logging=False).logger is None
        assert isinstance(LLMEpiParamGenerator(chat_llm("x")).logger.generator_dir, Path)

    def test_text_log_format_leaves_logger_undefined(self, chat_llm):
        # NOTE: current behavior — possible bug: enable_logging=True with log_format="text"
        # never assigns self.logger, so generate() dies with AttributeError
        gen = LLMEpiParamGenerator(chat_llm(params_json(0.1, 0.05, 0.005)), log_format="text")
        assert not hasattr(gen, "logger")
        with pytest.raises(AttributeError):
            gen.generate({"expert_comment": "x", "current_episode": make_episode(), "iteration": 0})

    def test_prompt_template_variables(self, generator):
        # NOTE: current_beta/current_gamma/current_mu/direction_hint are passed to the chain
        # but have no placeholder — they never reach the model
        assert set(generator.prompt.input_variables) == {
            "expected_position",
            "expected_height",
            "sensitivity_map",
            "task_config",
            "beta_min",
            "beta_max",
            "gamma_min",
            "gamma_max",
            "mu_min",
            "mu_max",
            "current_episode",
            "history",
            "expert_comment",
            "stats_summary",
            "target_metrics",
        }

    def test_balance_analysis_section_is_an_empty_header(self, generator):
        human = generator.prompt.messages[1].prompt.template
        after = human.split("## 7. BALANCE ANALYSIS (CRITICAL!)")[1]
        assert after.strip().startswith("Based on the history above, check if you are FOCUSING ONLY ON ONE METRIC:")
        assert "Generate new parameters" in after

    def test_set_task_config_and_update_history_print(self, generator, capsys):
        generator.set_task_config({"description": "demo"})
        generator.update_history([make_episode(), make_episode(iteration=1)])
        out = capsys.readouterr().out
        assert "✅ Task configured: demo" in out
        assert "📚 Updated generator history with 2 episodes" in out
        assert generator.task_config == {"description": "demo"} and len(generator.history) == 2


class TestGeneratorFormatting:
    """The helper methods that render history, statistics, task config and the sensitivity map into prompt text."""

    def test_history_empty(self, generator):
        assert generator._format_history([]) == "No previous attempts available."

    def test_history_uses_episode_prompt_format_and_keeps_the_last_five(self, generator):
        eps = [make_episode(iteration=i, timestamp="ts") for i in range(7)]
        text = generator._format_history(eps)
        assert "**Iteration 1:**" not in text
        assert "**Iteration 2:**" in text and "**Iteration 6:**" in text
        assert text.count("**Iteration") == 5
        assert text.endswith(eps[-1].to_prompt_format() + "\n")

    def test_history_fallback_for_foreign_objects(self, generator):
        foreign = types.SimpleNamespace(iteration=9, beta=0.1, gamma=0.05, mu=0.005, peak_position=10.0, peak_height=5.0, total_deaths=1.0, accepted=True, reasoning="R" * 150)
        text = generator._format_history([foreign])
        assert "Episode 9:" in text and "✅ ACCEPTED" in text
        assert "R" * 100 + "..." in text and "R" * 101 not in text

    def test_stats_empty(self, generator):
        assert generator._format_stats_summary([]) == "No statistics available yet."

    def test_stats_summary_ranges_and_best_attempt(self, generator):
        generator.task_config = {"target_peak": 100}
        eps = [
            make_episode(iteration=0, beta=0.09, peak_position=120.0, peak_height=9000, total_deaths=400),
            make_episode(iteration=1, beta=0.11, peak_position=104.0, peak_height=12000, total_deaths=600),
        ]
        text = generator._format_stats_summary(eps)
        assert "- β: 0.0900 – 0.1100" in text
        assert "- Peak position: 104.0 – 120.0 days (target: 100)" in text
        assert "- Total deaths: 400 – 600" in text
        assert "- β=0.1100, γ=0.0553, μ=0.00850" in text  # best = closest to target
        assert "Peak at day 104.0 with 12000 infected" in text

    def test_stats_summary_default_target_is_30(self, generator):
        text = generator._format_stats_summary([make_episode()])
        assert "(target: 30)" in text

    def test_stats_summary_without_metrics_crashes(self, generator):
        # NOTE: current behavior — min() on an empty peak list
        from formats.data_formats import Episode

        with pytest.raises(ValueError):
            generator._format_stats_summary([Episode(beta=0.1, gamma=0.05, mu=0.005, reasoning=None)])

    def test_stats_summary_foreign_objects(self, generator):
        assert generator._format_stats_summary([object()]) == "No valid parameter data in history."

    def test_current_episode(self, generator):
        assert generator._format_current_episode(None) == "No current episode available."
        text = generator._format_current_episode(make_episode())
        assert "- β = 0.0910" in text and "- Peak at day 120.0" in text and "- Total deaths: 500" in text
        assert generator._format_current_episode(object()).startswith("Unknown episode format: <class 'object'>")

    def test_task_config_formatting(self, generator):
        assert generator._format_task_config() == "No task configuration provided."
        generator.task_config = {"description": "d", "population": 6_000_000, "I0": 40, "target_peak": 90}
        text = generator._format_task_config()
        assert "- Description: d" in text and "- Population: 6,000,000" in text
        assert "- Initial infected: 40" in text and "- Target peak: 90 days" in text

    def test_task_config_without_population_crashes(self, generator):
        # NOTE: current behavior — possible bug: the default '1,000,000' is a string and
        # cannot be formatted with the ',' specifier
        generator.task_config = {"description": "d"}
        with pytest.raises(ValueError):
            generator._format_task_config()

    def test_target_metrics(self, generator):
        assert generator._format_target_metrics() == "No target metrics specified."
        generator.task_config = {"target_peak": 90, "target_height": 12345.6, "target_deaths": 78.9}
        text = generator._format_target_metrics()
        assert "- Desired peak position: 90 days" in text
        assert "- Desired peak height: 12,346 infected" in text
        assert "- Desired total deaths: 79" in text

    def test_sensitivity_map_empty(self, generator):
        assert generator._format_sensitivity_map({}) == "No sensitivity map available."

    def test_sensitivity_map_table(self, generator, sensitivity_map):
        text = generator._format_sensitivity_map(sensitivity_map)
        assert "Baseline: β=0.1000, γ=0.0600, μ=0.00300" in text
        assert "│ β        │ -2%" in text and "│ β        │ +2%" in text
        assert "│ γ        │ +1%" in text and "│ μ        │ -1%" in text
        assert text.count("│ β        │") == 4
        assert "combined" not in text.lower()  # combined variations are never shown to the model


# --------------------------------------------------------------------------- generator.generate


def _state(**overrides):
    state = {
        "expert_comment": "Need higher peak",
        "expected_position": "unchanged",
        "expected_height": "higher",
        "current_episode": make_episode(),
        "history": [make_episode()],
        "iteration": 0,
    }
    state.update(overrides)
    return state


def _logged_generator_prompt(tmp_path):
    files = list((tmp_path / "logs" / "prompts" / "generator").glob("generator_iter_*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


class TestGenerate:
    """LLMEpiParamGenerator.generate(): state validation, trust region, retries, logging and the returned state."""

    @pytest.mark.parametrize("comment", [None, "", "   "])
    def test_expert_comment_is_mandatory(self, generator, comment):
        with pytest.raises(ValueError, match="Expert comment is required"):
            generator.generate(_state(expert_comment=comment))

    def test_missing_current_episode_returns_state_untouched(self, generator, capsys):
        state = _state(current_episode=None)
        out = generator.generate(state)
        assert out is state and "generated_params" not in state
        assert "❌ No current episode in state" in capsys.readouterr().out

    def test_success_writes_params_and_increments_iteration(self, generator):
        state = _state()
        generator.generate(state)
        assert state["generated_params"] == {
            "reasoning": "scripted parameters",
            "beta": 0.095,
            "gamma": 0.0553,
            "mu": 0.0085,
            "confidence": "high",
        }
        assert state["iteration"] == 1

    def test_missing_iteration_key_crashes(self, generator):
        # NOTE: current behavior — the increment uses state['iteration'], not .get()
        state = _state()
        del state["iteration"]
        with pytest.raises(KeyError):
            generator.generate(state)

    def test_prompt_carries_direction_and_trust_region(self, generator, tmp_path):
        generator.generate(_state())
        log = _logged_generator_prompt(tmp_path)
        prompt = log["prompt"]
        assert "Expected position change: **unchanged**" in prompt
        assert "Expected height change: **higher**" in prompt
        assert "β (infection rate): between 0.08100 and 0.10100" in prompt
        assert "γ (recovery rate): between 0.05480 and 0.05580" in prompt
        assert "μ (mortality rate): between 0.008450 and 0.008550" in prompt
        assert "No sensitivity map available." in prompt
        assert "## 4. Expert Comment (Goal to Achieve)\nNeed higher peak" in prompt

    def test_trust_region_follows_the_current_episode_not_the_baseline(self, generator, tmp_path):
        generator.generate(_state(current_episode=make_episode(beta=0.2, gamma=0.1, mu=0.01)))
        prompt = _logged_generator_prompt(tmp_path)["prompt"]
        assert "between 0.19000 and 0.21000" in prompt
        assert "between 0.09950 and 0.10050" in prompt

    def test_sensitivity_map_is_rendered_into_the_prompt(self, generator, sensitivity_map, tmp_path):
        generator.generate(_state(sensitivity_map=sensitivity_map))
        prompt = _logged_generator_prompt(tmp_path)["prompt"]
        assert "PARAMETER SENSITIVITY MAP" in prompt and "│ β        │ +1%" in prompt

    def test_direction_hint_is_printed_but_not_prompted(self, generator, tmp_path, capsys):
        generator.generate(_state(direction_hint="raise beta gently"))
        assert "💡 Hint: raise beta gently" in capsys.readouterr().out
        assert "raise beta gently" not in _logged_generator_prompt(tmp_path)["prompt"]

    def test_log_payload(self, generator, tmp_path, config_env):
        config_env(MODEL_NAME_HF="org/configured-model")
        generator.set_task_config({"description": "d", "population": 1000, "I0": 1})
        generator.generate(_state(iteration=2))
        log = _logged_generator_prompt(tmp_path)
        assert log["iteration"] == 2
        assert log["parsed_output"]["beta"] == 0.095
        assert log["context"]["current_episode_beta"] == 0.091
        assert log["context"]["current_episode_accepted"] is True
        assert log["context"]["expert_comment"] == "Need higher peak"
        assert log["context"]["task_config"]["description"] == "d"
        assert log["metadata"]["iteration"] == 2
        # the generator passes model=getattr(llm, "model_id", "unknown"), but PromptLogger
        # merges config metadata on top, so the config model name wins
        assert log["metadata"]["model"] == "org/configured-model"
        assert log["metadata"]["provider"] == "huggingface"

    def test_no_logging_when_disabled(self, chat_llm, tmp_path):
        gen = LLMEpiParamGenerator(chat_llm(params_json(0.1, 0.05, 0.005)), enable_logging=False)
        gen.generate(_state())
        assert not (tmp_path / "logs").exists()

    def test_out_of_trust_region_values_are_not_rejected(self, chat_llm):
        # NOTE: current behavior — bounds live only in the prompt text; EpiParameters allows β up to 3.0
        gen = LLMEpiParamGenerator(chat_llm(params_json(2.0, 0.9, 0.09)))
        state = _state()
        gen.generate(state)
        assert state["generated_params"]["beta"] == 2.0

    def test_retry_recovers_from_a_bad_first_reply(self, chat_llm, tmp_path):
        gen = LLMEpiParamGenerator(chat_llm("not json", params_json(0.1, 0.05, 0.005)))
        state = _state()
        gen.generate(state)
        assert state["generated_params"]["beta"] == 0.1
        assert _logged_generator_prompt(tmp_path)["response_raw"] == "not json"

    def test_exhausted_retries_record_the_error(self, chat_llm, capsys):
        gen = LLMEpiParamGenerator(chat_llm("not json"))
        state = _state()
        out = gen.generate(state)
        assert out is state
        assert state["generated_params"] is None
        assert state["generation_error"].startswith("Parser error after 3 retries:")
        assert state["iteration"] == 0
        assert "❌ All retry attempts failed" in capsys.readouterr().out

    def test_wrapped_base_client_always_needs_one_retry(self, tmp_path):
        # NOTE: current behavior — possible bug: with a string-returning LLM (the pipeline's
        # LLMClientWrapper) `raw_response = str(result)` is the repr of the whole
        # {'response': ..., 'prompt': ...} dict, so the first parse fails and only the
        # retry (which reads the bare string) succeeds.
        client = ScriptedClient([params_json(0.1, 0.05, 0.005)])
        wrapper = LLMEpiParamGenerator(client)._create_langchain_compatible_llm()
        gen = LLMEpiParamGenerator(wrapper)
        state = _state()
        gen.generate(state)
        assert state["generated_params"]["beta"] == 0.1
        assert len(client.prompts) == 2
        assert _logged_generator_prompt(tmp_path)["response_raw"].startswith("{'response': '{")

    def test_generate_is_the_call_operator(self, generator):
        state = _state()
        assert generator(state) is state and "generated_params" in state

    def test_console_summary(self, generator, capsys):
        generator.generate(_state())
        out = capsys.readouterr().out
        assert "🎯 LLM-EPIPARAM GENERATOR AGENT" in out
        assert "🎯 Expected: position=unchanged, height=higher" in out
        assert "✅ Generated: β=0.0950, γ=0.0553, μ=0.00850" in out
        assert "💭 Reasoning: scripted parameters" in out
