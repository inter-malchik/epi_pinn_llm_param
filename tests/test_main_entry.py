"""main_test.main() — the entry point, driven end to end with a scripted LLM.

The CSV path inside main() points outside the repository and every PINN is
trained for 10 000 epochs, so the test swaps pd.read_csv for the synthetic
baseline dataset and shortens every training run to one epoch; everything
else (graph, surrogate, MC Dropout, reports, plots) is real.

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

import main_test
from agents.PINN_const import EINN_PINN
from tests.support import ScriptedClient, intent_json, params_json, route_by_prompt

pytestmark = pytest.mark.slow


@pytest.fixture
def entry_point(monkeypatch, synthetic_df):
    def prepare(params_reply):
        client = ScriptedClient(route_by_prompt(intent_json(False, "any", True, "higher"), params_reply))
        monkeypatch.setattr(main_test.pd, "read_csv", lambda path: synthetic_df.copy())
        monkeypatch.setattr(main_test.LLMFactory, "from_config", staticmethod(lambda cfg: client))
        original = EINN_PINN.train_model
        monkeypatch.setattr(EINN_PINN, "train_model", lambda self, n_epoch=1, **kw: original(self, n_epoch=1, **kw))
        return client

    return prepare


class TestMain:
    """End-to-end runs of main() with the accept path and the always-reject path."""

    def test_accept_path_produces_every_artifact(self, entry_point, tmp_path, capsys):
        client = entry_point(params_json(0.1, 0.06, 0.003))
        result, comparison = main_test.main()

        history = result["history"]
        assert [ep.iteration for ep in history] == [0, 1]
        assert history[1].accepted is True and history[1].beta == 0.1
        assert comparison["baseline_params_input"] == {"beta": 0.091, "gamma": 0.0553, "mu": 0.0085}
        assert comparison["optimized_params_input"] == {"beta": 0.1, "gamma": 0.06, "mu": 0.003}
        assert comparison["train_size"] == 120
        assert set(comparison["peak_analysis"]) == {"real_data", "baseline", "optimized", "changes"}

        cmp_dir = tmp_path / "PINN_comparison_results"
        assert len(list(cmp_dir.glob("comparison_*.png"))) == 1
        assert len(list(cmp_dir.glob("comparison_*.json"))) == 1
        assert len(list(cmp_dir.glob("peak_analysis_*.png"))) == 1
        assert len(list(cmp_dir.glob("summary_report_*.png"))) == 1
        assert len(list(cmp_dir.glob("full_comparison_*.png"))) == 1
        assert len(list((cmp_dir / "pdf_plots").glob("I_plot_*.pdf"))) == 1
        verification = list((tmp_path / "PINN_verification_results").glob("verification_*.json"))
        assert len(verification) == 1
        assert json.loads(verification[0].read_text())["train_size"] == 120
        assert len(list((tmp_path / "logs" / "prompts" / "generator").glob("generator_iter_*.json"))) == 1
        assert not (tmp_path / "pipeline_graph.png").exists()  # mermaid rendering needs the network

        out = capsys.readouterr().out
        assert "📌 Data mode: SYNTHETIC" in out
        assert "✅ Loaded 366 data points" in out
        assert "Max iterations: 5" in out  # printed value; run() actually gets 10
        assert "⚠️ pinn_verification not in result, trying to load from disk..." in out
        assert "✅ Loaded from disk" in out
        assert "✅ TEST COMPLETE" in out
        assert "📊 Full comparison plot:" in out
        assert len(client.prompts) == 3

    def test_no_acceptance_hits_the_langgraph_recursion_limit(self, entry_point, tmp_path):
        # NOTE: current behavior — possible bug: main() asks for max_iterations=10, but every
        # loop is 4 graph steps and the compiled graph keeps LangGraph's default
        # recursion_limit=25, so a run that keeps rejecting dies during the 6th iteration —
        # before the loop can end and before the `t_final` NameError in step 10 is reached
        from langgraph.errors import GraphRecursionError

        entry_point(params_json(0.085, 0.0553, 0.0085))  # lower β while the expert wants a higher peak
        with pytest.raises(GraphRecursionError):
            main_test.main()
        assert not (tmp_path / "PINN_verification_results").exists()
        assert len(list((tmp_path / "logs" / "prompts" / "generator").glob("generator_iter_*.json"))) == 6
