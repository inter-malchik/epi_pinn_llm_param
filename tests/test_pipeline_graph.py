"""main_test.OptimizationPipeline — construction, graph topology and end-to-end runs
through the compiled LangGraph with a scripted LLM.

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
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

import main_test
from agents.PINN_const import EINN_PINN
from agents.PINNAgent import PINNAgent
from agents.SurrogateModel import SurrogateAgent
from main_test import OptimizationPipeline, PINNNode, PINNVerificationNode, visualize_pipeline
from tests.support import ScriptedClient, intent_json, params_json, route_by_prompt
from tests.test_data_formats import DECLARED_STATE_KEYS

BASELINE = dict(beta=0.091, gamma=0.0553, mu=0.0085)


@pytest.fixture
def fast_verification(monkeypatch):
    """_build_graph hard-codes PINNVerificationNode(n_epoch=10000); train for 2 epochs instead."""
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return PINNVerificationNode(n_epoch=2, save_plots=kwargs.get("save_plots", True))

    monkeypatch.setattr(main_test, "PINNVerificationNode", factory)
    return seen


def _client(intent=intent_json(False, "any", True, "higher"), params=params_json(0.1, 0.06, 0.003)):
    return ScriptedClient(route_by_prompt(intent, params))


def _pipeline(client=None, **kwargs):
    pinn_agent = PINNAgent(pinn_class=EINN_PINN, n_epoch=1, device="cpu", verbose=False)
    return OptimizationPipeline(llm=client or _client(), pinn_agent=pinn_agent, surrogate_agent=SurrogateAgent(verbose=False), **kwargs)


# --------------------------------------------------------------------------- construction


class TestConstruction:
    """OptimizationPipeline.__init__: LLM wrapping, default agents, critic thresholds, PINN toggle."""

    def test_base_client_is_wrapped_for_every_agent(self, fast_verification):
        client = _client()
        p = _pipeline(client)
        assert p.is_base_client is True and p.llm is client
        wrapper = p.intent_parser.llm
        assert type(wrapper).__name__ == "LLMClientWrapper"
        assert p.generator.llm is wrapper and p.critic.llm is wrapper
        assert wrapper._llm_type == "base_llm_client_wrapper"
        assert wrapper._identifying_params == {"model": "scripted-model"}
        # the wrapper exposes the client's temperature read/write (RetryParser binds it)
        assert wrapper.temperature == 0.0
        wrapper.temperature = 0.5
        assert client.temperature == 0.5

    def test_langchain_llm_is_used_as_is(self, chat_llm, fast_verification):
        llm = chat_llm("x")
        p = OptimizationPipeline(llm=llm, use_pinn=False)
        assert p.is_base_client is False and p.intent_parser.llm is llm and p.generator.llm is llm

    def test_default_components(self, fast_verification):
        p = OptimizationPipeline(llm=_client())
        assert isinstance(p.surrogate_agent, SurrogateAgent) and p.surrogate_agent.verbose is True
        assert isinstance(p.pinn_agent, PINNAgent)
        assert p.pinn_agent.n_epoch == 10_000 and p.pinn_agent.lambda_data == 0.01 and p.pinn_agent.lambda_ode == 1.0
        assert p.pinn_agent.results_dir == "PINN_agent_results"
        assert isinstance(p.pinn_node, PINNNode)
        assert p.initial_conditions == {"population": 10_000, "S0": 9_999, "I0": 1, "R0": 0, "D0": 0}
        assert isinstance(p.memory, MemorySaver)
        assert p.use_pinn is True

    def test_critic_thresholds_are_the_documented_ones(self, fast_verification):
        c = _pipeline().critic
        assert (c.position_threshold, c.height_threshold_relative, c.position_tolerance, c.height_tolerance_relative) == (1.0, 0.03, 250.0, 0.03)
        assert c.logger is not None

    def test_verification_node_is_built_with_10000_epochs(self, fast_verification):
        _pipeline()
        assert fast_verification == {"n_epoch": 10000, "save_plots": True}

    def test_without_pinn(self, fast_verification):
        p = OptimizationPipeline(llm=_client(), use_pinn=False)
        assert p.pinn_agent is None and p.pinn_node is None
        assert fast_verification == {}

    def test_explicit_agents_are_kept(self, fast_verification):
        surrogate = SurrogateAgent(verbose=False)
        pinn_agent = PINNAgent(pinn_class=EINN_PINN, n_epoch=3)
        p = OptimizationPipeline(llm=_client(), surrogate_agent=surrogate, pinn_agent=pinn_agent, initial_conditions={"population": 5})
        assert p.surrogate_agent is surrogate and p.pinn_agent is pinn_agent and p.pinn_node.pinn_agent is pinn_agent
        assert p.initial_conditions == {"population": 5}


# --------------------------------------------------------------------------- topology


def _topology(pipeline):
    g = pipeline.graph.get_graph(xray=True)
    nodes = set(g.nodes)
    edges = {(e.source, e.target) for e in g.edges}
    conditional = {(e.source, e.target) for e in g.edges if e.conditional}
    return nodes, edges, conditional


class TestTopology:
    """Node/edge structure of the compiled LangGraph, with and without PINN."""

    def test_nodes_and_edges_with_pinn(self, fast_verification):
        nodes, edges, conditional = _topology(_pipeline())
        assert nodes == {"__start__", "sensitivity", "intent", "generate", "surrogate", "critic", "history", "pinn", "pinn_verification", "__end__"}
        assert edges == {
            ("__start__", "sensitivity"),
            ("sensitivity", "intent"),
            ("intent", "generate"),
            ("generate", "surrogate"),
            ("surrogate", "critic"),
            ("critic", "history"),
            ("history", "pinn_verification"),
            ("history", "generate"),
            ("history", "__end__"),
            ("pinn_verification", "__end__"),
        }
        assert conditional == {("history", "pinn_verification"), ("history", "generate"), ("history", "__end__")}

    def test_pinn_node_is_registered_but_unreachable(self, fast_verification):
        # NOTE: current behavior — possible bug: "pinn" has neither incoming nor outgoing edges
        nodes, edges, _ = _topology(_pipeline())
        assert "pinn" in nodes
        assert not [e for e in edges if e[1] == "pinn"]
        assert not [e for e in edges if e[0] == "pinn"]

    def test_nodes_and_edges_without_pinn(self, fast_verification):
        nodes, edges, conditional = _topology(OptimizationPipeline(llm=_client(), use_pinn=False))
        assert "pinn" not in nodes and "pinn_verification" not in nodes
        assert ("history", "generate") in edges and ("history", "__end__") in edges
        assert ("history", "pinn_verification") not in edges
        assert conditional == {("history", "generate"), ("history", "__end__")}

    def test_visualize_pipeline_fails_closed_offline(self, fast_verification, tmp_path, capsys):
        # draw_mermaid_png() calls the mermaid.ink web API — blocked by the test harness
        assert visualize_pipeline(_pipeline(), "graph.png") is False
        assert not (tmp_path / "graph.png").exists()
        assert "❌ Error generating visualization" in capsys.readouterr().out


# --------------------------------------------------------------------------- run(): guards


class TestRunGuards:
    """Argument validation of run()."""

    def test_pinn_data_is_mandatory(self, fast_verification):
        with pytest.raises(ValueError, match="pinn_data is required"):
            _pipeline().run(**BASELINE, expert_comment="x")

    def test_pinn_data_must_be_non_empty(self, fast_verification):
        with pytest.raises(ValueError, match="non-empty"):
            _pipeline().run(**BASELINE, expert_comment="x", pinn_data={"S": [], "I": [], "R": [], "D": []})


# --------------------------------------------------------------------------- run(): accept path


def _generator_prompts(tmp_path):
    files = sorted((tmp_path / "logs" / "prompts" / "generator").glob("generator_iter_*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


class TestRunAcceptPath:
    """A full run where the first proposal is accepted: history, state keys that survive the graph, Phase-3 artifacts."""

    @pytest.fixture
    def outcome(self, fast_verification, pinn_data_small, tmp_path):
        client = _client()
        pipeline = _pipeline(client)
        final = pipeline.run(**BASELINE, expert_comment="Need higher peak", max_iterations=5, t_max=200, num_points=200, pinn_data=pinn_data_small)
        return pipeline, client, final

    def test_accepted_on_the_first_iteration(self, outcome):
        pipeline, client, final = outcome
        history = final["history"]
        assert [ep.iteration for ep in history] == [0, 1]
        assert history[0].accepted is True and history[0].expert_comment == "BASELINE: Need higher peak"
        assert history[1].accepted is True and (history[1].beta, history[1].gamma, history[1].mu) == (0.1, 0.06, 0.003)
        assert history[1].peak_height > history[0].peak_height
        assert final["critic_decision"] == "accept" and final["iteration"] == 1
        assert final["current_episode"] == history[1]
        assert final["should_continue"] is True

    def test_critic_and_history_node_build_two_different_episodes(self, outcome):
        # NOTE: current behavior — possible bug: the critic numbers its episode
        # len(critic.history)+1 (= 2 after the baseline) while HistoryNode uses
        # state['iteration'] (= 1); same parameters, two objects, two numbers
        _, _, final = outcome
        critic_episode, history_episode = final["final_episode"], final["history"][1]
        assert critic_episode is not history_episode
        assert critic_episode.iteration == 2 and history_episode.iteration == 1
        assert (critic_episode.beta, critic_episode.peak_position) == (history_episode.beta, history_episode.peak_position)

    def test_agents_share_one_history_that_contains_both_copies(self, outcome):
        # NOTE: current behavior — critique() appends its own episode, then HistoryNode adds
        # another one (the de-duplication compares iteration numbers, which differ)
        pipeline, _, final = outcome
        assert pipeline.critic.history is pipeline.generator.history
        assert [ep.iteration for ep in pipeline.critic.history] == [0, 2, 1]
        assert [ep.iteration for ep in final["history"]] == [0, 1]

    def test_undeclared_keys_do_not_survive_the_graph(self, outcome, tmp_path):
        # NOTE: current behavior — possible bug: LangGraph keeps only PipelineState channels,
        # so sensitivity_map / pinn_verification / initial_conditions / simulation_params are
        # gone from the final state, and the generator never saw the sensitivity map
        _, _, final = outcome
        assert set(final) <= DECLARED_STATE_KEYS
        for key in ("sensitivity_map", "pinn_verification", "initial_conditions", "simulation_params", "expert_intent"):
            assert key not in final
        prompts = _generator_prompts(tmp_path)
        assert len(prompts) == 1
        assert "No sensitivity map available." in prompts[0]["prompt"]

    def test_expert_intent_reached_the_generator_prompt(self, outcome, tmp_path):
        prompt = _generator_prompts(tmp_path)[0]["prompt"]
        assert "Expected position change: **unchanged**" in prompt
        assert "Expected height change: **higher**" in prompt
        assert "between 0.08100 and 0.10100" in prompt  # trust region around the baseline

    def test_llm_was_asked_twice(self, outcome):
        _, client, _ = outcome
        # intent parser + generator; the generator's first parse fails on the wrapper path,
        # so the retry adds a third prompt — see test_agents_intent_generator
        assert len(client.prompts) == 3
        assert client.prompts[0].startswith("System: You are an epidemiologist.")
        assert "SIRD model parameter optimization" in client.prompts[1]
        assert client.prompts[2] == client.prompts[1]

    def test_phase_3_ran_and_left_its_report_on_disk(self, outcome, tmp_path):
        reports = list((tmp_path / "PINN_verification_results").glob("verification_*.json"))
        assert len(reports) == 1
        data = json.loads(reports[0].read_text())
        assert data["success"] is True
        assert data["llm_params"] == {"beta": 0.1, "gamma": 0.06, "mu": 0.003, "reasoning": "scripted parameters", "confidence": "high"}
        assert data["train_size"] == 30
        assert len(list((tmp_path / "PINN_verification_plots").glob("*.png"))) == 2

    def test_surrogate_results_belong_to_the_accepted_parameters(self, outcome):
        _, _, final = outcome
        assert final["surrogate_results"]["beta"] == 0.1
        assert final["surrogate_results"]["peak_position"] == final["history"][1].peak_position
        assert final["task_config"]["baseline_peak"] == final["history"][0].peak_position
        assert final["task_config"]["pinn_data"]["train_size"] == 30

    def test_console_summary(self, fast_verification, pinn_data_small, capsys):
        _pipeline().run(**BASELINE, expert_comment="Need higher peak", max_iterations=3, t_max=200, num_points=200, pinn_data=pinn_data_small)
        out = capsys.readouterr().out
        assert "🚀 STARTING OPTIMIZATION PIPELINE" in out
        assert "Population: 1000" in out and "S0=999, I0=1, R0=0, D0=0" in out
        assert "➡️  ACCEPTED → Running PINN VERIFICATION (Phase 3)" in out
        assert "🏁 PIPELINE COMPLETE" in out
        assert "✅ BEST OPTIMIZED (Iteration 1):" in out
        assert out.count("🏁 PIPELINE COMPLETE") == 2  # the summary block is printed twice


# --------------------------------------------------------------------------- run(): reject path


class TestRunRejectPath:
    """Runs where every proposal is rejected: iteration counting, prompt history, recursion limit."""

    def test_loop_until_max_iterations(self, fast_verification, pinn_data_small, tmp_path):
        # the generator lowers β while the expert wants a higher peak → always rejected
        pipeline = _pipeline(_client(params=params_json(0.085, 0.0553, 0.0085)))
        final = pipeline.run(**BASELINE, expert_comment="Need higher peak", max_iterations=2, t_max=200, num_points=200, pinn_data=pinn_data_small)
        history = final["history"]
        # NOTE: current behavior — route_after_history increments a copy of the state, so the
        # counter advances exactly once per loop (in the generator): no double increment
        assert [ep.iteration for ep in history] == [0, 1, 2]
        assert [ep.accepted for ep in history] == [True, False, False]
        assert final["iteration"] == 2 and final["critic_decision"] == "reject"
        assert all("expected higher" in ep.reasoning for ep in history[1:])
        assert len(_generator_prompts(tmp_path)) == 2
        assert not (tmp_path / "PINN_verification_results").exists()

    def test_second_iteration_sees_the_first_attempt(self, fast_verification, pinn_data_small, tmp_path):
        _pipeline(_client(params=params_json(0.085, 0.0553, 0.0085))).run(**BASELINE, expert_comment="Need higher peak", max_iterations=2, t_max=200, num_points=200, pinn_data=pinn_data_small)
        second = _generator_prompts(tmp_path)[1]["prompt"]
        assert "**Iteration 1:**" in second and "✗ REJECTED" in second
        assert "between 0.07500 and 0.09500" in second  # trust region moved with the rejected attempt

    def test_more_than_five_rejections_exceed_the_recursion_limit(self, fast_verification, pinn_data_small, tmp_path):
        # NOTE: current behavior — possible bug: the graph is compiled without a recursion_limit
        # and run() does not pass one, so LangGraph's default of 25 steps (2 + 4 per loop)
        # aborts the 6th iteration — max_iterations above 5 can never be reached
        from langgraph.errors import GraphRecursionError

        pipeline = _pipeline(_client(params=params_json(0.085, 0.0553, 0.0085)))
        with pytest.raises(GraphRecursionError):
            pipeline.run(**BASELINE, expert_comment="Need higher peak", max_iterations=10, t_max=200, num_points=200, pinn_data=pinn_data_small)
        assert len(_generator_prompts(tmp_path)) == 6
        assert [ep.iteration for ep in pipeline.critic.history if ep.iteration <= 5 and ep.accepted is False]  # rejected attempts were recorded before the abort

    def test_console_reports_no_acceptance(self, fast_verification, pinn_data_small, capsys):
        _pipeline(_client(params=params_json(0.085, 0.0553, 0.0085))).run(**BASELINE, expert_comment="Need higher peak", max_iterations=1, t_max=200, num_points=200, pinn_data=pinn_data_small)
        out = capsys.readouterr().out
        assert "⚠️ No episodes were accepted during optimization" in out
        assert "⏹️  Max iterations reached → END" in out

    def test_without_pinn_the_decision_node_drives_the_loop(self, pinn_data_small, capsys):
        pipeline = OptimizationPipeline(llm=_client(params=params_json(0.085, 0.0553, 0.0085)), use_pinn=False, surrogate_agent=SurrogateAgent(verbose=False))
        final = pipeline.run(**BASELINE, expert_comment="Need higher peak", max_iterations=2, t_max=200, num_points=200, pinn_data=pinn_data_small)
        assert [ep.iteration for ep in final["history"]] == [0, 1, 2]
        assert "⚖️ DECISION NODE" in capsys.readouterr().out

    def test_without_pinn_acceptance_ends_immediately(self, pinn_data_small):
        pipeline = OptimizationPipeline(llm=_client(), use_pinn=False, surrogate_agent=SurrogateAgent(verbose=False))
        final = pipeline.run(**BASELINE, expert_comment="Need higher peak", max_iterations=5, t_max=200, num_points=200, pinn_data=pinn_data_small)
        assert [ep.accepted for ep in final["history"]] == [True, True]
        assert "pinn_results" not in final


# --------------------------------------------------------------------------- run(): failures


class TestRunFailures:
    """How run() dies when the LLM output cannot be parsed."""

    def test_generator_that_never_yields_json_crashes_in_the_critic(self, fast_verification, pinn_data_small):
        # NOTE: current behavior — possible bug: after 3 failed parses generated_params is None;
        # SurrogateNode skips quietly, then the critic indexes new_params['beta'] on None
        pipeline = _pipeline(_client(params="I refuse to answer in JSON"))
        with pytest.raises(TypeError, match="not subscriptable"):
            pipeline.run(**BASELINE, expert_comment="Need higher peak", max_iterations=2, t_max=200, num_points=200, pinn_data=pinn_data_small)

    def test_unparseable_intent_stops_the_graph(self, fast_verification, pinn_data_small):
        from langchain_core.exceptions import OutputParserException

        pipeline = _pipeline(_client(intent="no json"))
        with pytest.raises(OutputParserException):
            pipeline.run(**BASELINE, expert_comment="Need higher peak", max_iterations=2, t_max=200, num_points=200, pinn_data=pinn_data_small)


# --------------------------------------------------------------------------- run(): baseline failure & graph export


class TestRunBaselineAndExport:
    """run() aborts when the baseline surrogate cannot be computed; visualize_pipeline
    writes the PNG only when the mermaid renderer succeeds."""

    def test_baseline_simulation_failure_is_a_runtime_error(self, fast_verification, pinn_data_small):
        pipeline = _pipeline()
        # the baseline is computed through pipeline.surrogate_node; make that step fail
        pipeline.surrogate_node = lambda state: {**state, "surrogate_results": {"success": False, "error": "boom"}}
        with pytest.raises(RuntimeError, match="Failed to compute baseline: boom"):
            pipeline.run(**BASELINE, expert_comment="x", pinn_data=pinn_data_small)

    def test_t_max_and_num_points_given_to_run_are_never_used(self, fast_verification, pinn_data_small, initial_conditions):
        # NOTE: current behavior — possible bug: run() keeps t_max/num_points only in
        # task_config and in the undeclared 'simulation_params' key. The baseline_state has no
        # 'simulation_params' at all, and inside the graph the key is dropped (see
        # test_undeclared_keys_do_not_survive_the_graph), so baseline AND every candidate are
        # simulated on the SurrogateAgent defaults: 400 days, 1000 points.
        pipeline = _pipeline(_client(params=params_json(0.085, 0.0553, 0.0085)))
        final = pipeline.run(**BASELINE, expert_comment="Need higher peak", max_iterations=1, t_max=50, num_points=20, pinn_data=pinn_data_small)
        baseline, candidate = final["history"]
        surrogate = SurrogateAgent(verbose=False)
        reference_baseline = surrogate.simulate(**BASELINE, **initial_conditions, t_max=400, num_points=1000)
        reference_candidate = surrogate.simulate(beta=0.085, gamma=0.0553, mu=0.0085, **initial_conditions, t_max=400, num_points=1000)
        assert baseline.peak_position == reference_baseline["peak_position"] > 50
        assert candidate.peak_position == reference_candidate["peak_position"] > 50
        assert len(final["surrogate_results"]["t"]) == 1000
        assert final["task_config"]["t_max"] == 50 and final["task_config"]["num_points"] == 20  # stored, unused

    def test_visualize_pipeline_writes_the_png_when_rendering_works(self, fast_verification, tmp_path, capsys):
        pipeline = _pipeline()

        class FakeDrawing:
            def draw_mermaid_png(self):
                return b"\x89PNG fake"

        class FakeGraph:
            def get_graph(self, xray=False):
                assert xray is True
                return FakeDrawing()

        pipeline.graph = FakeGraph()
        assert visualize_pipeline(pipeline, "graph.png") is True
        assert (tmp_path / "graph.png").read_bytes() == b"\x89PNG fake"
        assert "✅ Pipeline visualization saved to graph.png" in capsys.readouterr().out
