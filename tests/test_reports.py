"""main_test.py — run_pinn_comparison, create_summary_report, create_comparison_plot.

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

import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes

from agents.PINN_const import EINN_PINN
from main_test import create_comparison_plot, create_summary_report, run_pinn_comparison
from tests.support import make_episode

BASELINE = {"beta": 0.091, "gamma": 0.0553, "mu": 0.0085}
OPTIMIZED = {"beta": 0.1, "gamma": 0.06, "mu": 0.003}


@pytest.fixture
def capture_text(monkeypatch):
    """Record every string drawn with Axes.text so report contents can be asserted."""
    seen = []
    original = Axes.text

    def spy(self, x, y, s, *args, **kwargs):
        seen.append(s)
        return original(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", spy)
    return seen


# --------------------------------------------------------------------------- run_pinn_comparison


class TestRunPinnComparison:
    """run_pinn_comparison(): trains baseline and optimized PINNs, reuse of pipeline predictions, artifacts."""

    def test_trains_two_models_and_writes_artifacts(self, pinn_data_small, tmp_path):
        res = run_pinn_comparison(BASELINE, OPTIMIZED, pinn_data_small, results_dir="cmp", n_epoch=2)
        assert set(res) == {
            "timestamp", "baseline_params_input", "optimized_params_input", "baseline_pinn_estimated",
            "optimized_pinn_estimated", "baseline_pinn", "optimized_pinn", "peak_analysis", "plots",
            "t_data", "I_real", "i_predictions", "train_size",
        }
        assert res["baseline_params_input"] is BASELINE and res["optimized_params_input"] is OPTIMIZED
        assert res["baseline_pinn_estimated"] == pytest.approx(BASELINE, abs=1e-6)  # frozen parameters
        assert res["optimized_pinn_estimated"] == pytest.approx(OPTIMIZED, abs=1e-6)
        assert len(res["baseline_pinn"]["losses"]) == 2 and len(res["optimized_pinn"]["losses"]) == 2
        assert res["train_size"] == 30 and len(res["t_data"]) == 40 and len(res["I_real"]) == 40
        assert len(res["i_predictions"]["baseline"]) == 40
        assert set(res["peak_analysis"]) == {"real_data", "baseline", "optimized", "changes"}
        assert set(res["peak_analysis"]["changes"]) == {"peak_day", "peak_height", "error_to_real_day"}
        out = tmp_path / "cmp"
        names = sorted(p.name for p in out.iterdir())
        ts = res["timestamp"]
        assert names == [f"comparison_{ts}.json", f"comparison_{ts}.png", f"peak_analysis_{ts}.png"]
        assert res["plots"] == [str(out.relative_to(tmp_path) / f"comparison_{ts}.png"), str(out.relative_to(tmp_path) / f"peak_analysis_{ts}.png")]

    def test_json_on_disk_predates_the_trajectory_fields(self, pinn_data_small, tmp_path):
        # NOTE: current behavior — t_data / I_real / i_predictions / train_size are added to the
        # returned dict only after the JSON has been written
        res = run_pinn_comparison(BASELINE, OPTIMIZED, pinn_data_small, results_dir="cmp", n_epoch=1)
        data = json.loads((tmp_path / "cmp" / f"comparison_{res['timestamp']}.json").read_text())
        assert "t_data" not in data and "i_predictions" not in data
        assert data["peak_analysis"] == res["peak_analysis"]

    def test_real_data_peak_comes_from_the_input(self, pinn_data_small):
        res = run_pinn_comparison(BASELINE, OPTIMIZED, pinn_data_small, results_dir="cmp", n_epoch=1)
        I = np.array(pinn_data_small["I"])
        assert res["peak_analysis"]["real_data"] == {"day": float(np.argmax(I)), "height": float(I.max())}

    def test_pipeline_predictions_are_reused(self, pinn_data_small, monkeypatch):
        calls = []
        original = EINN_PINN.train_model
        monkeypatch.setattr(EINN_PINN, "train_model", lambda self, **kw: calls.append(kw) or original(self, **kw))
        n = len(pinn_data_small["I"])
        pipeline_results = {
            "success": True,
            "predictions": {"I": [1.0] * n, "S": [2.0] * n, "R": [3.0] * n, "D": [4.0] * n},
            "estimated_params": {"beta": 0.5, "gamma": 0.5, "mu": 0.05},
            "losses": [9.0],
        }
        res = run_pinn_comparison(BASELINE, OPTIMIZED, pinn_data_small, pipeline_pinn_results=pipeline_results, results_dir="cmp", n_epoch=1)
        assert len(calls) == 1  # only the baseline was trained
        assert res["optimized_pinn_estimated"] == {"beta": 0.5, "gamma": 0.5, "mu": 0.05}
        assert res["optimized_pinn"]["losses"] == [9.0]
        assert res["i_predictions"]["optimized"] == [1.0] * n

    def test_pipeline_results_without_predictions_trigger_training(self, pinn_data_small, monkeypatch):
        calls = []
        original = EINN_PINN.train_model
        monkeypatch.setattr(EINN_PINN, "train_model", lambda self, **kw: calls.append(kw) or original(self, **kw))
        res = run_pinn_comparison(BASELINE, OPTIMIZED, pinn_data_small, pipeline_pinn_results={"success": True}, results_dir="cmp", n_epoch=1)
        assert len(calls) == 2
        assert res["optimized_pinn_estimated"] == pytest.approx(OPTIMIZED, abs=1e-6)

    def test_failed_pipeline_results_trigger_training(self, pinn_data_small, monkeypatch):
        calls = []
        original = EINN_PINN.train_model
        monkeypatch.setattr(EINN_PINN, "train_model", lambda self, **kw: calls.append(kw) or original(self, **kw))
        run_pinn_comparison(BASELINE, OPTIMIZED, pinn_data_small, pipeline_pinn_results={"success": False}, results_dir="cmp", n_epoch=1)
        assert len(calls) == 2

    def test_loss_weights_are_forwarded(self, pinn_data_small, monkeypatch):
        calls = []
        original = EINN_PINN.train_model
        monkeypatch.setattr(EINN_PINN, "train_model", lambda self, **kw: calls.append(kw) or original(self, **kw))
        run_pinn_comparison(BASELINE, OPTIMIZED, pinn_data_small, results_dir="cmp", n_epoch=3, lambda_data=0.5, lambda_ode=2.0, lambda_ic=0.3, lambda_bc=0.7)
        assert calls == [{"n_epoch": 3, "lambda_data": 0.5, "lambda_ode": 2.0, "lambda_ic": 0.3, "lambda_bc": 0.7}] * 2

    def test_defaults(self):
        import inspect

        params = inspect.signature(run_pinn_comparison).parameters
        assert params["results_dir"].default == "PINN_comparison_results"
        assert params["n_epoch"].default == 5_000
        assert (params["lambda_data"].default, params["lambda_ode"].default, params["lambda_ic"].default, params["lambda_bc"].default) == (1.0, 1.0, 0.1, 0.0)

    def test_console_summary(self, pinn_data_small, capsys):
        run_pinn_comparison(BASELINE, OPTIMIZED, pinn_data_small, results_dir="cmp", n_epoch=1)
        out = capsys.readouterr().out
        assert "🧠 PINN COMPARISON: Baseline vs Optimized Predictions" in out
        assert "📊 STEP 1: Training PINN for BASELINE parameters" in out
        assert "📋 PINN VALIDATION SUMMARY" in out


# --------------------------------------------------------------------------- create_summary_report


def _comparison(day_change=5.0, height_change=100.0):
    t = list(range(40))
    return {
        "baseline_params_input": BASELINE,
        "optimized_params_input": OPTIMIZED,
        "baseline_pinn_estimated": BASELINE,
        "optimized_pinn_estimated": OPTIMIZED,
        "peak_analysis": {
            "real_data": {"day": 18.0, "height": 900.0},
            "baseline": {"day": 20.0, "height": 1000.0},
            "optimized": {"day": 20.0 + day_change, "height": 1000.0 + height_change},
            "changes": {"peak_day": day_change, "peak_height": height_change, "error_to_real_day": 0.0},
        },
        "t_data": t,
        "I_real": [float(i) for i in t],
        "i_predictions": {"baseline": [float(i) for i in t], "optimized": [float(i) + 1 for i in t]},
        "train_size": 30,
    }


def _report(comparison, output_path, **kw):
    history = [make_episode(), make_episode(iteration=1, accepted=False), make_episode(iteration=2)]
    return create_summary_report(
        comparison_results=comparison, history=history, baseline_episode=history[0], optimized_episode=history[2],
        expert_comment="Need higher peak", n_epoch=10, lambda_data=1.0, lambda_ode=1.0, lambda_ic=0.1, lambda_bc=0.0,
        output_path=output_path, **kw,
    )


class TestCreateSummaryReport:
    """create_summary_report(): the one-image report, its success criterion and text formatting."""

    def test_writes_the_png(self, tmp_path, capsys):
        path = _report(_comparison(), str(tmp_path / "report.png"))
        assert path == str(tmp_path / "report.png") and Path(path).stat().st_size > 0
        assert "📄 Summary report saved:" in capsys.readouterr().out

    def test_default_path_needs_an_existing_results_dir(self, tmp_path):
        # NOTE: current behavior — the default location is not created by this function
        with pytest.raises(FileNotFoundError):
            _report(_comparison(), None)
        (tmp_path / "PINN_comparison_results").mkdir()
        path = _report(_comparison(), None)
        assert path.startswith("PINN_comparison_results/summary_report_") and Path(path).is_file()

    def test_any_nonzero_change_counts_as_success(self, tmp_path, capture_text):
        # NOTE: current behavior — possible bug: direction is never compared with the expert's
        # request; a peak that went UP under "Need lower peak" is still reported as a success
        _report(_comparison(day_change=-3.0, height_change=+250.0), str(tmp_path / "r.png"))
        summary = next(s for s in capture_text if "SUMMARY" in s)
        assert "✅ Optimization successfully changed peak characteristics" in summary
        # NOTE: current behavior — possible bug: the format spec `:+>8.1f` uses '+' as the
        # FILL character (not the sign flag), so numbers are padded with plus signs
        assert "Peak Day Change:     ++++-3.0 days" in summary
        assert "Peak Height Change:  +++++250 infected" in summary
        assert "• Peak day:     20 → 17 days" in summary
        assert "• Peak height:  1000 → 1250 infected" in summary

    def test_zero_change_is_flagged(self, tmp_path, capture_text):
        _report(_comparison(day_change=0.0, height_change=0.0), str(tmp_path / "r.png"))
        summary = next(s for s in capture_text if "SUMMARY" in s)
        assert "⚠️ No significant changes in peak characteristics" in summary

    def test_parameter_table_and_training_block(self, tmp_path, capture_text):
        _report(_comparison(), str(tmp_path / "r.png"))
        params = next(s for s in capture_text if "Number of epochs" in s)
        assert "Number of epochs:           10" in params
        assert "λ_bc (boundary cond):    0.000" in params
        table = next(s for s in capture_text if "║ Param" in s)
        assert "║ β       ║ 0.0910        ║ 0.1000        ║ +0.0090          ║" in table
        assert "║ PINN μ  ║ 0.00850       ║ 0.00300       ║ -0.00550         ║" in table
        assert any("Expert comment: 'Need higher peak'" in s for s in capture_text)


# --------------------------------------------------------------------------- create_comparison_plot


def _plot_inputs(with_ci=True):
    t_real = np.arange(60, dtype=float)
    I_real = 100 * np.exp(-((t_real - 25) ** 2) / 80)
    t_grid = np.linspace(0, 100, 120)
    I_base = 90 * np.exp(-((t_grid - 30) ** 2) / 90)
    I_synth = 110 * np.exp(-((t_grid - 22) ** 2) / 70)
    I_final = 105 * np.exp(-((t_grid - 24) ** 2) / 75)
    final = {"t": t_grid, "I": I_final, "beta": 0.1, "gamma": 0.06, "mu": 0.003}
    if with_ci:
        final["ci_lower"] = I_final * 0.9
        final["ci_upper"] = I_final * 1.1
    return (
        {"t": t_real, "I": I_real, "S": 1000 - I_real, "R": I_real * 0, "D": I_real * 0},
        {"t": t_grid, "I": I_base, **BASELINE},
        {"t": t_grid, "I": I_synth, **OPTIMIZED},
        final,
    )


class TestCreateComparisonPlot:
    """create_comparison_plot(): PNG + PDF output, confidence band smoothing, peak analysis text."""

    def test_writes_png_and_pdf(self, tmp_path, capsys):
        real, base, synth, final = _plot_inputs()
        path = create_comparison_plot(real, base, synth, final, "Need higher peak", train_split_time=40, output_path=str(tmp_path / "full.png"))
        assert path == str(tmp_path / "full.png") and Path(path).stat().st_size > 0
        pdfs = list((tmp_path / "PINN_comparison_results" / "pdf_plots").glob("I_plot_*.pdf"))
        assert len(pdfs) == 1 and pdfs[0].stat().st_size > 0
        out = capsys.readouterr().out
        assert "✅ PDF saved:" in out and "✅ Full comparison plot saved:" in out

    def test_default_output_path_creates_the_results_dir(self, tmp_path):
        real, base, synth, final = _plot_inputs()
        path = create_comparison_plot(real, base, synth, final, "x")
        assert path.startswith("PINN_comparison_results/full_comparison_") and Path(path).is_file()

    def test_without_ci_and_without_split(self, tmp_path, capture_text):
        real, base, synth, final = _plot_inputs(with_ci=False)
        create_comparison_plot(real, base, synth, final, "x", output_path=str(tmp_path / "p.png"))
        analysis = next(s for s in capture_text if "PEAK ANALYSIS" in s)
        assert "95% CI width at peak" not in analysis

    def test_peak_analysis_text(self, tmp_path, capture_text):
        real, base, synth, final = _plot_inputs()
        create_comparison_plot(real, base, synth, final, "x", train_split_time=40, output_path=str(tmp_path / "p.png"), ci_smoothing_sigma=1.0)
        analysis = next(s for s in capture_text if "PEAK ANALYSIS" in s)
        assert "Initial → Synthetic: Δβ = +0.0090" in analysis
        assert "Initial → Final:     Δβ = +0.0090" in analysis
        assert "Synthetic → Final:   Δβ = +0.0000" in analysis
        assert "95% CI width at peak" in analysis
        assert "Real data            25.0         100" in analysis

    def test_ci_smoothing_uses_the_given_sigma(self, tmp_path, monkeypatch):
        import scipy.ndimage

        sigmas = []
        original = scipy.ndimage.gaussian_filter1d
        monkeypatch.setattr(scipy.ndimage, "gaussian_filter1d", lambda arr, sigma: sigmas.append(sigma) or original(arr, sigma))
        real, base, synth, final = _plot_inputs()
        create_comparison_plot(real, base, synth, final, "x", output_path=str(tmp_path / "p.png"), ci_smoothing_sigma=7.5)
        assert sigmas == [7.5, 7.5]

    def test_default_sigma_is_two(self):
        import inspect

        assert inspect.signature(create_comparison_plot).parameters["ci_smoothing_sigma"].default == 2.0

    def test_style_is_switched_globally(self, tmp_path):
        real, base, synth, final = _plot_inputs()
        create_comparison_plot(real, base, synth, final, "x", output_path=str(tmp_path / "p.png"))
        assert matplotlib.rcParams["axes.grid"] is True  # seaborn-v0_8-whitegrid stays active
